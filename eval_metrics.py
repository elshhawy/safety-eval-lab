import csv, hashlib

import sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
THRS = [round(0.05 * i, 2) for i in range(1, 20)]
VEST_COLS = ["vest", "safety_vest", "high_visibility_vest", "reflective_vest"]


def is_dev(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10 < 3


def score(r, cfg):
    if r["matched"] != "1":
        return None
    if cfg == "max_vest":
        return max(float(r[c]) for c in VEST_COLS)
    return float(r[cfg])


def counts(sub, item, cfg, thr):
    gt_col = "gt_helmet" if item == "helmet" else "gt_vest"
    c = dict(tp=0, fp=0, fn=0, tn=0, unk=0, n=0, total_no=0)
    for r in sub:
        g = r[gt_col]
        if g == "unknown":
            c["unk"] += 1
            continue
        c["n"] += 1
        if g == "no":
            c["total_no"] += 1
        s = score(r, cfg)
        if s is None:
            continue
        missing = s < thr
        if g == "no" and missing:
            c["tp"] += 1
        elif g == "yes" and missing:
            c["fp"] += 1
        elif g == "no" and not missing:
            c["fn"] += 1
        else:
            c["tn"] += 1
    return c


def prf(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    vp = tp / (tp + fp) if tp + fp else None
    vr = tp / (tp + fn) if tp + fn else None
    e2e = tp / c["total_no"] if c["total_no"] else None
    wp = tn / (tn + fn) if tn + fn else None
    wr = tn / (tn + fp) if tn + fp else None
    return vp, vr, e2e, wp, wr


def f2(p, rc):
    if p is None or rc is None or (4 * p + rc) == 0:
        return 0.0
    return 5 * p * rc / (4 * p + rc)


def f(x):
    return " -  " if x is None else f"{x:.2f}"


rows = list(csv.DictReader(open(f"{RES}/persons.csv")))
VEST_COLS = [c for c in VEST_COLS if c in rows[0]]
dev = [r for r in rows if is_dev(r["image"])]
fin = [r for r in rows if not is_dev(r["image"])]
print(f"persons in labels: dev={len(dev)} final={len(fin)}")

imgs = list(csv.DictReader(open(f"{RES}/images.csv")))
for name, flt in (("dev", True), ("final", False)):
    s = [r for r in imgs if is_dev(r["image"]) == flt]
    g = sum(int(r["n_gt_person"]) for r in s)
    p = sum(int(r["n_pred_person"]) for r in s)
    m = sum(int(r["n_matched"]) for r in s)
    print(f"person detector [{name}]: GT={g} pred={p} matched={m} recall={m / g:.2f} precision={m / p:.2f}")

for item, cfgs in (("helmet", ["helmet"]), ("vest", VEST_COLS + ["max_vest"])):
    print(f"\n=== {item} ===")
    best = None
    for cfg in cfgs:
        cfg_best = None
        for thr in THRS:
            vp, vr = prf(counts(dev, item, cfg, thr))[:2]
            sc = f2(vp, vr)
            if cfg_best is None or sc > cfg_best[0]:
                cfg_best = (sc, thr)
        print(f"  dev best per prompt: {cfg:22s} thr={cfg_best[1]:.2f} F2={cfg_best[0]:.2f}")
        if best is None or cfg_best[0] > best[0]:
            best = (cfg_best[0], cfg, cfg_best[1])
    _, cfg, thr = best
    print(f"  CHOSEN on dev: prompt={cfg} thr={thr:.2f}  (person is 'missing' when score < thr)")
    groups = (
        ("dev", dev),
        ("final", fin),
        ("final/source", [r for r in fin if r["prefix"] == "source"]),
        ("final/ultralytics", [r for r in fin if r["prefix"] == "ultralytics"]),
    )
    for name, sub in groups:
        c = counts(sub, item, cfg, thr)
        vp, vr, e2e, wp, wr = prf(c)
        print(f"  {name:18s} n={c['n']:4d} unknown={c['unk']:3d} | missing-detect P={f(vp)} R(matched)={f(vr)} R(end-to-end)={f(e2e)} | wearing P={f(wp)} R={f(wr)}")
