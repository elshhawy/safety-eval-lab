"""PPE scoring interface: one person crop in, helmet and vest scores out.

    scorer = PPEScorer()                      # default: the provisional champion
    scores = scorer.score(crop_bgr)           # {"helmet": 0.0..1.0, "vest": 0.0..1.0}
    flags = is_violation(scores, {"helmet": 0.30, "vest": 0.05})

The model is chosen by name from MODELS, or with the environment variable PPE_MODEL.
To add another model, add an entry to MODELS (pin repo, file, revision and sha256).

Protocol (the same as the evaluation in compare_metrics.py):
- the input is a BGR crop of ONE person, not the full frame;
- the PPE model runs on the crop at conf=0.001;
- the score of an item is the MAXIMUM confidence among detections of its class,
  and 0.0 when there is none;
- an item is a violation when its score is BELOW its threshold.

Thresholds are NOT part of the model. The values in REFERENCE_THRESHOLDS were chosen
on dev splits of two public datasets. A real camera needs its own calibration.

Smoke test on one image (the image is treated as a crop):
    cd work && python ../ppe_model.py <image.jpg>
"""
import hashlib
import os
import sys

ITEMS = ("helmet", "vest")
DEFAULT_MODEL = "safetyvision-v2"

MODELS = {
    "safetyvision-v2": {
        "repo": "ayushgupta7777/safetyvision-yolov8",
        "file": "v2/best.pt",
        "revision": "56a71758b55f0e9f2b4b2d6b51a779a1f882da10",
        "sha256": "7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a",
        "classes": {"Hardhat": "helmet", "Safety Vest": "vest"},
        "conf": 0.001,
    },
}

REFERENCE_THRESHOLDS = {
    "construction-ppe": {"helmet": 0.30, "vest": 0.05},
    "kaggle-ppe": {"helmet": 0.35, "vest": 0.10},
}


def scores_from_detections(detections, class_map):
    """detections: iterable of (class_name, confidence). Returns {item: max confidence}."""
    scores = {item: 0.0 for item in ITEMS}
    for name, conf in detections:
        item = class_map.get(name)
        if item is not None and conf > scores[item]:
            scores[item] = float(conf)
    return scores


def is_violation(scores, thresholds):
    """True for an item whose score is below its threshold."""
    return {item: scores[item] < thresholds[item] for item in ITEMS}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PPEScorer:
    def __init__(self, name=None):
        name = name or os.environ.get("PPE_MODEL", DEFAULT_MODEL)
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; known: {sorted(MODELS)}")
        self.name = name
        self.cfg = MODELS[name]
        self._model = self._load()

    def _load(self):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        c = self.cfg
        try:
            path = hf_hub_download(c["repo"], c["file"], revision=c["revision"],
                                   local_files_only=True)
        except Exception:
            path = hf_hub_download(c["repo"], c["file"], revision=c["revision"])
        if c.get("sha256") and _sha256(path) != c["sha256"]:
            raise RuntimeError(f"sha256 of {c['file']} does not match the pinned value")
        return YOLO(path)

    def score(self, crop):
        if crop is None or getattr(crop, "size", 0) == 0:
            raise ValueError("empty crop")
        r = self._model.predict(crop, conf=self.cfg["conf"], verbose=False)[0]
        detections = [(r.names[int(c)], float(p))
                      for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        return scores_from_detections(detections, self.cfg["classes"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python ../ppe_model.py <image.jpg>")
    import cv2

    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit(f"cannot read image: {sys.argv[1]}")
    scorer = PPEScorer()
    print("model:", scorer.name)
    print("scores:", {k: round(v, 3) for k, v in scorer.score(img).items()})
