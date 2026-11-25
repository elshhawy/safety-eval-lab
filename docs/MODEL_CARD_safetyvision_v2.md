# Model Card: safetyvision v2 (PPE model, provisional champion)

**Status: PROVISIONAL.** Selected on 2026-11-25 as the working PPE model. Not approved by the supervisor. Acceptance thresholds below are **proposed, temporary**.

## 1. Identity

| Field | Value |
|---|---|
| Source repo | `ayushgupta7777/safetyvision-yolov8` (Hugging Face) |
| File used | `v2/best.pt` |
| HF revision | `56a71758b55f0e9f2b4b2d6b51a779a1f882da10` |
| Size | 22,547,434 bytes |
| sha256 | `7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a` |
| Architecture | YOLOv8s (per Handoff_v7) |
| Classes used | `Hardhat`, `Safety Vest` (the `Person` class is not used; `yolo26s.pt` detects people) |
| Training data | Model page: fine-tuned from Ultralytics YOLOv8 on about 80k images (Albumentations augmentation). Our MLflow tag records it as 5 merged Roboflow datasets. We did not verify either. Overlap with our eval sets is unknown. |
| Fine-tuning by us | None. Measured as released. |
| License | The model page states the base model (Ultralytics YOLOv8) is AGPL-3.0. The repo's own license field was not confirmed. Treat as AGPL-3.0 until confirmed. |
| Ultralytics version | 8.4.216 |

The measured file is inferred to be this one: `eval_infer_hf.py` downloads `v2/best.pt` from this repo, and the local cache holds only this file. The sha256 was computed after the evaluation, so the script did not record it at run time.

## 2. Role in the pipeline

`frame -> person detector (yolo26s.pt, class 0, conf 0.25) -> crop per person -> safetyvision v2 -> helmet score and vest score -> compare with threshold -> violator / compliant / unknown`

- The model runs on a **crop of each person**, not on the full frame. Changing this changes the numbers.
- Person detector detections (`yolo26s.pt`, class 0, conf 0.25) are matched greedily to the ground-truth `Person` boxes at IoU 0.5. A ground-truth person with no match counts as missed.
- The PPE model runs on the crop at `conf=0.001`. The score of an item is the **maximum confidence** among its detections of that class (`Hardhat`, `Safety Vest`). The threshold is applied to this score.
- Positive class for all metrics: **violator** (no helmet / no vest).
- Cases with an unknown ground-truth state are **excluded** from the metrics (counts below).

## 3. Evaluation protocol

| Item | Construction-PPE (cppe) | Kaggle PPE |
|---|---|---|
| Dataset version | `construction-ppe-e5817aa9` | `kaggle-ppe-ndomalau-4060img` |
| Dev / final split | dev = md5(image name) % 10 < 3, final = the rest (1000 images) | dev = valid split, final = test split (812 images used) |
| Thresholds (chosen on dev) | helmet 0.30, vest 0.05 | helmet 0.35, vest 0.10 |
| Match IoU | 0.5 | 0.5 |
| PPE imgsz | 640 (Ultralytics default; the model card suggests 896) | 640 |
| Code | git commit `38bff95` (clean tree), `compare_metrics.py` | same |
| MLflow run id | `6dd9c48e54284bf4bc1cb2c91496fe40` | `78e76620de3943f399fd961debf106e7` |

MLflow experiment: `ppe-model-selection` (local `work/mlflow.db`).

## 4. Results (final split, as logged in MLflow)

Definitions (from `compare_metrics.py`; positive class = violator):

- **Recall** = caught violators / all violators. It is end to end: a violator the person detector never found counts as missed.
- **False alarm** = false alarms / compliant people (fp / (fp + tn)).
- **Precision** = real alarms / all alarms (tp / (tp + fp)). It depends on how many violators the dataset contains, so do not compare it across datasets.
- **Intervals**: Wilson 95% (z = 1.96) for Recall and False alarm. Precision has no interval.

**Construction-PPE (final, 1000 images)**

| Item | Recall | False alarm | Precision |
|---|---|---|---|
| Helmet | 0.8442 [0.8105, 0.8729] | 0.0808 [0.0658, 0.0990] | 0.8394 |
| Vest | 0.8763 [0.8398, 0.9054] | 0.0918 [0.0766, 0.1095] | 0.7572 |

**Kaggle PPE (test, 812 images)**

| Item | Recall | False alarm | Precision |
|---|---|---|---|
| Helmet | 0.8697 [0.8319, 0.9000] | 0.1964 [0.1623, 0.2357] | 0.7880 |
| Vest | 0.8873 [0.8565, 0.9122] | 0.2248 [0.1840, 0.2716] | 0.8497 |

**Counts**

| Dataset | Item | Violators | Compliant | Unknown (excluded) |
|---|---|---|---|---|
| cppe | Helmet | 520 | 1071 | 105 |
| cppe | Vest | 388 | 1245 | 63 |
| Kaggle | Helmet | 376 | 496 | 82 |
| Kaggle | Vest | 497 | 379 | 78 |

**Person detector (yolo26s, as logged).** The last two columns are the share of violators lost because the person detector did not find them. This loss is already included in Recall above.

| Dataset | Person precision | Person recall | Helmet violators missed (person not detected) | Vest violators missed (person not detected) |
|---|---|---|---|---|
| cppe | 0.9703 | 0.9257 | 0.1038 | 0.1005 |
| Kaggle | 0.8544 | 0.8857 | 0.0665 | 0.0825 |

**Author-reported (model page, not ours; different data and protocol, do not compare directly):** held-out test mAP@0.5 = 0.754 at imgsz 640 and 0.766 at imgsz 896; validation mAP@0.5 = 0.787.

## 5. Why it is the provisional champion

- Recall is close among the shortlisted models; differences are within the intervals.
- Its False alarm is the lowest on both datasets, for both helmet and vest.
- Speed (FPS) was **not measured**, so it is not part of the decision.

## 6. Limitations and caveats

- **Higher False alarm on Kaggle** (about 20% helmet, 22% vest) than on cppe (about 8% / 9%). Domain shift is visible.
- **Vest Precision on cppe is 0.7572**, lower than the other cells.
- **Part of the missed violators comes from the person detector, not from this model.** About 10% of violators on cppe and 7 to 8% on Kaggle are never found as a person.
- **Thresholds are tied to these datasets.** A real camera needs its own calibration. The cppe vest threshold (0.05) sits at the edge of the search grid, so it is fragile.
- **Train/eval overlap is unknown.** A filename-pattern check on Kaggle found 9/3248 (train), 4/406 (valid), 0/406 (test) matching names. This checks names only and does not prove there is no overlap.
- **Run at imgsz 640**, while the model card suggests 896. Results at 896 are not measured.
- **Not compared against any fine-tuned model.** Fine-tuning is planned for Phase 5.
- **Unknown cases are excluded**, so the metrics describe only cases with a known state.
- **AGPL-3.0** obligations apply to anything that distributes or serves this model.

## 7. Proposed acceptance criteria (TEMPORARY, not approved)

- Minimum Recall: TODO (to be set with the supervisor)
- Maximum False alarm: TODO (to be set with the supervisor)

## 8. Reproduce

1. Activate the venv: `source /e/dev/safety-eval-lab/.venv/Scripts/activate`
2. From `work/`: `python ../log_mlflow_models.py` (reads `metrics_summary.csv`, logs 10 runs)
3. Verify weights: `sha256sum` of `v2/best.pt` must equal the hash in section 1.

## 9. Open items

- Confirm the repo's own license field on the model page (the base model is stated as AGPL-3.0).
- Measure FPS on the target CPU.
- Calibrate thresholds on the target camera.
- Supervisor to set acceptance criteria (section 7).
