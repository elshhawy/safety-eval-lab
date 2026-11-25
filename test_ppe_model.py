"""Tests for ppe_model.py that need no weights, no network and no ultralytics.

Run from the repo root or from work/:
    python ../test_ppe_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ppe_model as pm

CLASSES = pm.MODELS[pm.DEFAULT_MODEL]["classes"]


def test_max_per_class():
    dets = [("Hardhat", 0.2), ("Hardhat", 0.7), ("Safety Vest", 0.4)]
    assert pm.scores_from_detections(dets, CLASSES) == {"helmet": 0.7, "vest": 0.4}


def test_missing_item_is_zero():
    assert pm.scores_from_detections([("Hardhat", 0.5)], CLASSES) == {"helmet": 0.5, "vest": 0.0}
    assert pm.scores_from_detections([], CLASSES) == {"helmet": 0.0, "vest": 0.0}


def test_unmapped_classes_are_ignored():
    dets = [("Person", 0.99), ("Gloves", 0.9)]
    assert pm.scores_from_detections(dets, CLASSES) == {"helmet": 0.0, "vest": 0.0}


def test_violation_rule_is_strictly_below_threshold():
    thr = {"helmet": 0.30, "vest": 0.05}
    assert pm.is_violation({"helmet": 0.29, "vest": 0.04}, thr) == {"helmet": True, "vest": True}
    assert pm.is_violation({"helmet": 0.30, "vest": 0.05}, thr) == {"helmet": False, "vest": False}


def test_unknown_model_name_is_rejected():
    try:
        pm.PPEScorer("does-not-exist")
    except ValueError as e:
        assert "unknown model" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_pinned_config_is_complete():
    cfg = pm.MODELS[pm.DEFAULT_MODEL]
    assert len(cfg["revision"]) == 40 and len(cfg["sha256"]) == 64
    assert set(cfg["classes"].values()) == set(pm.ITEMS)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
