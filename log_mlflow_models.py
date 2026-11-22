import csv, datetime, hashlib, os, subprocess, sys
from importlib.metadata import PackageNotFoundError, version
import mlflow
from mlflow.tracking import MlflowClient

HERE = os.path.dirname(os.path.abspath(__file__))
AUTHOR = os.environ.get("MLFLOW_AUTHOR", "elshhawy")
DATE = datetime.date.today().isoformat()
EXPERIMENT = "ppe-model-selection"

DATASETS = {
    "cppe": ("construction-ppe-e5817aa9", "dev=md5(image name)%10<3, final=the rest (1000 images)"),
    "kaggle": ("kaggle-ppe-ndomalau-4060img", "dev=valid split, final=test split (812 images used)"),
}
# key, ppe model description, overlap note, results folder per dataset
MODELS = [
    ("yoloe-11s-seg", "yoloe-11s-seg.pt (text prompts)",
     "zero-shot, not trained on these datasets",
     {"cppe": "results", "kaggle": "results_kaggle"}),
    ("iaac-yolo26s", "iaac-maai/day4-example-ppe-detector model.pt",
     "trained on Roboflow Construction Site Safety; overlap with the Kaggle set unknown",
     {"cppe": "results_iaac", "kaggle": "results_iaac_kaggle"}),
    ("harsh77", "harsh-77/ppe-detection best.pt",
     "training data not documented; overlap unknown",
     {"cppe": "results_harsh77_cppe", "kaggle": "results_harsh77_kaggle"}),
    ("safetyvision-v2", "ayushgupta7777/safetyvision-yolov8 v2/best.pt",
     "trained on 5 merged Roboflow datasets; overlap unknown; run at imgsz 640 (card suggests 896)",
     {"cppe": "results_safetyvision_cppe", "kaggle": "results_safetyvision_kaggle"}),
    ("constructelligence-v0.3", "constructelligence/construction-site-safety-hazards buildvision-hazards-v0.3.pt",
     "training data unknown; author evaluated on an older Construction-PPE version; overlap unknown",
     {"cppe": "results_constructelligence_cppe", "kaggle": "results_constructelligence_kaggle"}),
]
FIELDS = ["recall", "recall_lo", "recall_hi", "false_alarm", "false_alarm_lo", "false_alarm_hi",
          "precision", "missed_by_person_detector", "violators", "compliant", "unknown"]


def is_dev_md5(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10 < 3


def person_final(folder, dkey):
    imgs = list(csv.DictReader(open(f"{folder}/images.csv")))
    if dkey == "kaggle":
        sel = [r for r in imgs if r["split"] == "test"]
    else:
        sel = [r for r in imgs if not is_dev_md5(r["image"])]
    g = sum(int(r["n_gt_person"]) for r in sel)
    p = sum(int(r["n_pred_person"]) for r in sel)
    m = sum(int(r["n_matched"]) for r in sel)
    return (m / g if g else None), (m / p if p else None)


def num(x):
    return float(x) if x not in ("", None) else None


if os.path.basename(os.getcwd()) != "work":
    sys.exit("GUARD FAILED: run this from the work/ folder (cd work first). Nothing was logged.")
if not os.path.exists("metrics_summary.csv"):
    sys.exit("GUARD FAILED: metrics_summary.csv not found. Run compare_metrics.py on all 10 folders first.")

summary = list(csv.DictReader(open("metrics_summary.csv")))
plan = []
for key, desc, note, folders in MODELS:
    for dkey, folder in folders.items():
        rows = {r["item"]: r for r in summary if r["folder"] == folder}
        if set(rows) != {"helmet", "vest"}:
            sys.exit(f"GUARD FAILED: metrics_summary.csv has no helmet+vest rows for {folder}. Nothing was logged.")
        if not os.path.exists(f"{folder}/images.csv"):
            sys.exit(f"GUARD FAILED: {folder}/images.csv missing. Nothing was logged.")
        plan.append((key, desc, note, dkey, folder, rows))

commit = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(
    ["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"], text=True).strip())
try:
    ul_version = version("ultralytics")
except PackageNotFoundError:
    ul_version = "unknown"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
exp = mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

for key, desc, note, dkey, folder, rows in plan:
    dataset_version, split_rule = DATASETS[dkey]
    eval_key = f"{key}|{dkey}"
    old = client.search_runs([exp.experiment_id], filter_string=f"tags.eval_key = '{eval_key}'")
    for o in old:
        client.delete_run(o.info.run_id)
    metrics, params = {}, {}
    for item in ("helmet", "vest"):
        r = rows[item]
        params[f"{item}_prompt"] = r["prompt"]
        params[f"{item}_threshold"] = r["threshold"]
        for f in FIELDS:
            v = num(r[f])
            if v is not None:
                metrics[f"{item}_{f}"] = v
    pr, pp = person_final(folder, dkey)
    if pr is not None:
        metrics["person_final_recall"] = pr
    if pp is not None:
        metrics["person_final_precision"] = pp
    with mlflow.start_run(run_name=f"{AUTHOR}_{key}_{dkey}_{DATE}"):
        mlflow.set_tags({"author": AUTHOR, "git_commit": commit, "dataset_version": dataset_version,
                         "git_dirty": str(dirty), "eval_key": eval_key, "candidate": key,
                         "dataset": dkey, "overlap_note": note})
        mlflow.log_params({
            "person_model": "yolo26s.pt", "person_conf": 0.25, "iou_match": 0.5,
            "ppe_model": desc, "ppe_imgsz": "640 (ultralytics default)",
            "split_rule": split_rule, "unknown_policy": "excluded from metrics",
            "metrics_script": "compare_metrics.py", "ultralytics_version": ul_version,
            **params})
        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(f"{k},{v}" for k, v in sorted({**params, **metrics}.items())), "run_summary.csv")
    print(f"{eval_key}: logged {len(metrics)} metrics (replaced {len(old)} older run(s)) | "
          f"helmet recall={metrics.get('helmet_recall', 0):.2f} false_alarm={metrics.get('helmet_false_alarm', 0):.2f} | "
          f"vest recall={metrics.get('vest_recall', 0):.2f} false_alarm={metrics.get('vest_false_alarm', 0):.2f}")
