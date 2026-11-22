import sys, os, glob, csv, time
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

BASE = "E:/dev/data-work/raw/Personal-Protective-Equipment (PPE) Dataset"
OUT = "results_iaac_kaggle"
PERSON_CONF = 0.25
IOU_MATCH = 0.5
C_HELMET, C_VEST, C_PERSON, C_NOHELMET, C_NOVEST = 0, 4, 3, 1, 2
WANT = {"helmet": "Hardhat", "vest": "Safety Vest", "nohelmet": "NO-Hardhat", "novest": "NO-Safety Vest"}


def read_labels(path, w, h):
    boxes = {}
    for line in open(path):
        t = line.split()
        if len(t) < 5:
            continue
        c = int(t[0])
        cx, cy, bw, bh = [float(v) for v in t[1:5]]
        box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        boxes.setdefault(c, []).append(box)
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside(box, pbox):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return pbox[0] <= cx <= pbox[2] and pbox[1] <= cy <= pbox[3]


def status(yes_boxes, no_boxes, pbox):
    y = sum(inside(b, pbox) for b in yes_boxes)
    n = sum(inside(b, pbox) for b in no_boxes)
    if y > 0 and n == 0:
        return "yes"
    if n > 0 and y == 0:
        return "no"
    return "unknown"


limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
os.makedirs(OUT, exist_ok=True)
person_model = YOLO("yolo26s.pt")
ppe_model = YOLO(hf_hub_download("iaac-maai/day4-example-ppe-detector", "model.pt"))
names = set(ppe_model.names.values())
print("candidate classes:", sorted(names))
missing = [v for v in WANT.values() if v not in names]
if missing:
    sys.exit(f"class names not found: {missing}")

items = []
for split in ("valid", "test"):
    for p in sorted(glob.glob(f"{BASE}/{split}/images/*.jpg")):
        items.append((split, p))
if limit:
    step = max(1, len(items) // limit)
    items = items[::step][:limit]
print("images to process:", len(items))

f1 = open(f"{OUT}/persons.csv", "w", newline="")
w1 = csv.writer(f1)
w1.writerow(["image", "prefix", "split", "gt_helmet", "gt_vest", "matched"] + list(WANT))
f2 = open(f"{OUT}/images.csv", "w", newline="")
w2 = csv.writer(f2)
w2.writerow(["image", "prefix", "split", "n_gt_person", "n_pred_person", "n_matched"])

tot_gt = tot_matched = 0
t0 = time.time()
for k, (split, path) in enumerate(items):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    prefix = "kaggle"
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    lab = read_labels(f"{BASE}/{split}/labels/{stem}.txt", w, h)
    gt_persons = lab.get(C_PERSON, [])
    res = person_model.predict(img, classes=[0], conf=PERSON_CONF, verbose=False)[0]
    preds = [tuple(float(v) for v in b.xyxy[0].tolist()) for b in res.boxes]

    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt_persons) for pi, p in enumerate(preds)), reverse=True)
    used_g, used_p, match = set(), set(), {}
    for v, gi, pi in pairs:
        if v < IOU_MATCH:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        match[gi] = pi

    scores_by_pred = {}
    for gi, pbox in enumerate(gt_persons):
        gh = status(lab.get(C_HELMET, []), lab.get(C_NOHELMET, []), pbox)
        gv = status(lab.get(C_VEST, []), lab.get(C_NOVEST, []), pbox)
        if gi in match:
            pi = match[gi]
            if pi not in scores_by_pred:
                x1, y1, x2, y2 = preds[pi]
                x1, y1 = max(int(x1), 0), max(int(y1), 0)
                x2, y2 = min(int(x2), w), min(int(y2), h)
                crop = img[y1:y2, x1:x2]
                best = {}
                if crop.size > 0:
                    r = ppe_model.predict(crop, conf=0.001, verbose=False)[0]
                    for pb in r.boxes:
                        c = r.names[int(pb.cls)]
                        best[c] = max(best.get(c, 0.0), float(pb.conf))
                scores_by_pred[pi] = [best.get(WANT[key], 0.0) for key in WANT]
            sc = ["%.4f" % v for v in scores_by_pred[pi]]
            matched = 1
        else:
            sc = [""] * len(WANT)
            matched = 0
        w1.writerow([name, prefix, split, gh, gv, matched] + sc)
    w2.writerow([name, prefix, split, len(gt_persons), len(preds), len(match)])
    tot_gt += len(gt_persons)
    tot_matched += len(match)
    if (k + 1) % 25 == 0:
        f1.flush()
        f2.flush()
        print(f"{k + 1}/{len(items)} done, {time.time() - t0:.0f}s elapsed", flush=True)

f1.close()
f2.close()
print(f"DONE: {len(items)} images, GT persons={tot_gt}, matched={tot_matched}, time={time.time() - t0:.0f}s")
