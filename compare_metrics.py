import csv, hashlib, math, os, sys

THRS = [round(0.05 * i, 2) for i in range(1, 20)]
VEST_ALL = ["vest", "safety_vest", "high_visibility_vest", "reflective_vest"]


def is_dev_md5(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10 < 3


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def score(r, cfg, vcols):
    if r["matched"] != "1":
        return None
    if cfg == "max_vest":
        return max(float(r[c]) for c in vcols)
    return float(r[cfg])


def counts(sub, item, cfg, thr, vcols):
    gt_col = "gt_helmet" if item == "helmet" else "gt_vest"
    c = dict(tp=0, fp=0, fn=0, tn=0, viol=0, comp=0, unk=0)
    for r in sub:
        g = r[gt_col]
        if g == "unknown":
            c["unk"] += 1
            continue
        c["viol" if g == "no" else "comp"] += 1
        s = score(r, cfg, vcols)
        if s is None:
            continue
        flagged = s < thr
        if g == "no":
            c["tp" if flagged else "fn"] += 1
        else:
            c["fp" if flagged else "tn"] += 1
    return c


def f2_on_dev(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0


def pct(x):
    return "  -  " if x is None else f"{100 * x:4.0f}%"


def rng(lo, hi):
    return "" if lo is None else f"[{100 * lo:.0f}-{100 * hi:.0f}%]"


summary = []
folders = sys.argv[1:]
if not folders:
    sys.exit("usage: python compare_metrics.py <results_folder> [<results_folder> ...]")

for folder in folders:
    rows = list(csv.DictReader(open(f"{folder}/persons.csv")))
    imgs = list(csv.DictReader(open(f"{folder}/images.csv")))
    if not rows:
        print(f"\n== {folder}: persons.csv is empty, skipped")
        continue
    vcols = [c for c in VEST_ALL if c in rows[0]]
    if any(r["split"] == "test" for r in rows):
        mode = "dev=valid, final=test"
        dev = [r for r in rows if r["split"] == "valid"]
        fin = [r for r in rows if r["split"] == "test"]
        fin_imgs = [r for r in imgs if r["split"] == "test"]
    else:
        mode = "dev=30% md5, final=70%"
        dev = [r for r in rows if is_dev_md5(r["image"])]
        fin = [r for r in rows if not is_dev_md5(r["image"])]
        fin_imgs = [r for r in imgs if not is_dev_md5(r["image"])]
    g = sum(int(r["n_gt_person"]) for r in fin_imgs)
    p = sum(int(r["n_pred_person"]) for r in fin_imgs)
    m = sum(int(r["n_matched"]) for r in fin_imgs)
    print(f"\n== {folder}  [{mode}]")
    if g and p:
        print(f"   person detector (final): found {m} of {g} real persons ({100 * m / g:.0f}%), {100 * m / p:.0f}% of its boxes were real")
    for item in ("helmet", "vest"):
        cfgs = ["helmet"] if item == "helmet" else vcols + (["max_vest"] if len(vcols) > 1 else [])
        best = None
        for cfg in cfgs:
            for thr in THRS:
                sc = f2_on_dev(counts(dev, item, cfg, thr, vcols))
                if best is None or sc > best[0]:
                    best = (sc, cfg, thr)
        _, cfg, thr = best
        c = counts(fin, item, cfg, thr, vcols)
        viol, comp = c["viol"], c["comp"]
        rec = c["tp"] / viol if viol else None
        far = c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else None
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        lost = (viol - c["tp"] - c["fn"]) / viol if viol else None
        rlo, rhi = wilson(c["tp"], viol)
        flo, fhi = wilson(c["fp"], c["fp"] + c["tn"])
        print(f"   --- {item}: prompt={cfg}, threshold={thr:.2f} (chosen on dev), final: {viol} violators, {comp} compliant, {c['unk']} unknown")
        print(f"       Recall      {pct(rec)} {rng(rlo, rhi):10s} caught {c['tp']} of {viol} violators")
        print(f"       False alarm {pct(far)} {rng(flo, fhi):10s} flagged {c['fp']} of {c['fp'] + c['tn']} compliant persons")
        print(f"       Precision   {pct(prec)}            {c['tp']} of {c['tp'] + c['fp']} alarms were real (depends on how many violators the data has)")
        print(f"       Missed because the person detector did not find them: {pct(lost)} of violators")
        summary.append(dict(folder=folder, mode=mode, item=item, prompt=cfg, threshold=thr,
                            violators=viol, compliant=comp, unknown=c["unk"],
                            recall=rec, recall_lo=rlo, recall_hi=rhi,
                            false_alarm=far, false_alarm_lo=flo, false_alarm_hi=fhi,
                            precision=prec, missed_by_person_detector=lost))

if summary:
    with open("metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("\nwrote metrics_summary.csv with", len(summary), "rows")
