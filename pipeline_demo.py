import sys, glob, os
import cv2
from ultralytics import YOLO, YOLOE

ROOT = "E:/dev/data-work/raw/extracted/construction-ppe/images/val"
OUT = "out"
PERSON_CONF = 0.4
PPE_CONF = 0.1
os.makedirs(OUT, exist_ok=True)

person_model = YOLO("yolo26s.pt")
ppe_model = YOLOE("yoloe-11s-seg.pt")
ppe_model.set_classes(["helmet", "vest"])

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
imgs = sorted(glob.glob(ROOT + "/*.jpg"))[:n]
print("images:", len(imgs))

for path in imgs:
    img = cv2.imread(path)
    name = os.path.basename(path)
    res = person_model.predict(img, classes=[0], conf=PERSON_CONF, verbose=False)[0]
    print(name, "persons:", len(res.boxes))
    for i, b in enumerate(res.boxes):
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        r = ppe_model.predict(crop, conf=PPE_CONF, verbose=False)[0]
        found = {}
        for pb in r.boxes:
            c = r.names[int(pb.cls)]
            found[c] = max(found.get(c, 0.0), float(pb.conf))
        label = " ".join(f"{k}:{found.get(k, 0.0):.2f}" for k in ("helmet", "vest"))
        print("   person", i, label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(f"{OUT}/{name}", img)
print("saved to work/out")
