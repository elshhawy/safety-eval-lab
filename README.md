# safety-eval-lab

Evaluation lab for the PPE part (helmet and vest) of an industrial safety monitoring graduation project. It measures candidate PPE models on the same protocol and logs the results in MLflow.

**Status: work in progress.** The current working model is **safetyvision v2**, chosen as a *provisional* champion. It is not approved by the project supervisor and it is not production ready. See [docs/MODEL_CARD_safetyvision_v2.md](docs/MODEL_CARD_safetyvision_v2.md).

## What is here

| File | What it does |
|---|---|
| `eval_infer.py`, `eval_infer_iaac.py` | Old scripts: the two baseline candidates on the Construction-PPE images |
| `eval_infer_kaggle.py`, `eval_infer_iaac_kaggle.py` | Same two candidates on the Kaggle dataset |
| `eval_infer_hf.py` | Hugging Face candidates. Usage: `python ../eval_infer_hf.py <harsh77\|safetyvision\|constructelligence> <cppe\|kaggle> [limit]` |
| `eval_metrics.py`, `eval_metrics_kaggle.py` | Old metric reports |
| `compare_metrics.py` | Recall, False alarm and Precision with 95% Wilson intervals. Usage: `python ../compare_metrics.py <results folder> ...` |
| `log_mlflow.py` | Old logger (experiment `ppe-baseline-eval`) |
| `log_mlflow_models.py` | Logs 10 runs (5 models on 2 datasets) in experiment `ppe-model-selection` |
| `register_champion.py` | Registers the provisional champion in the local MLflow registry with alias `champion` |
| `ppe_model.py` | Fixed interface: one person crop in, helmet and vest scores out. The model is chosen by name; the default is the provisional champion, pinned by revision and sha256 |
| `test_ppe_model.py` | Tests for the interface logic (no weights needed): `python test_ppe_model.py` |
| `pipeline_demo.py` | Early demo: person detector, then YOLOE on each person crop, saves annotated images (not the champion) |
| `requirements-lock.txt` | Package versions of the lab environment |

Scripts are meant to run from a local `work/` folder: `cd work && python ../script.py`.

## What is not here

Model weights, datasets, images and the MLflow database. They live in a git-ignored `work/` folder and in the Hugging Face cache on the author's machine. Reasons: size and licences. To rerun anything you need to download the datasets and models yourself.

## Metrics

Positive class = violator (no helmet, no vest).

- **Recall** = caught violators / all violators. A violator the person detector never found counts as missed.
- **False alarm** = false alarms / compliant people.
- **Precision** = real alarms / all alarms. It depends on how many violators the dataset contains, so compare Recall and False alarm across datasets, not Precision.

## Datasets (not included)

- **Construction-PPE**: "Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0". 1,000 images used as a frozen external test, never for training. Images were collected and generated, then annotated automatically and reviewed.
- **Kaggle "Personal-Protective-Equipment (PPE) Dataset"** by ndomalau: 4,060 images, CC BY 4.0 on the dataset page. The image sources are not stated and look mixed (stock photos, screenshots, video frames), so the licence may not cover each image.

## Licences

- This repository has no LICENSE file yet. The code licence is not decided.
- The evaluated models are Ultralytics YOLO based. Check each model page before reuse (the safetyvision base model is stated as AGPL-3.0).

## Limitations

- Test sets are public images of mixed origin, not footage from a real site.
- Thresholds were chosen per dataset. A real camera needs its own calibration.
- Whether the evaluated models saw the test images during training is unknown.
- Speed (FPS) has not been measured.
