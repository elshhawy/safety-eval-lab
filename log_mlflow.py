import os, re, subprocess, sys, datetime
from importlib.metadata import version
import mlflow

HERE = os.path.dirname(os.path.abspath(__file__))
AUTHOR = os.environ.get("MLFLOW_AUTHOR", "elshhawy")
DATASET_VERSION = "construction-ppe-e5817aa9"
DATE = datetime.date.today().isoformat()

RUNS = [
    {"name": "yoloe-11s-seg", "results": "results",
     "ppe_model": "yoloe-11s-seg.pt (text prompts)",
     "ppe_prompts": "helmet|vest|safety vest|high visibility vest|reflective vest"},
    {"name": "iaac-yolo26s", "results": "results_iaac",
     "ppe_model": "iaac-maai/day4-example-ppe-detector model.pt",
     "ppe_prompts": "classes Hardhat, Safety Vest"},
]

commit = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(
    ["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"], text=True).strip())

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ppe-baseline-eval")

KEYS = ["missing_precision", "missing_recall_matched", "missing_recall_e2e",
        "wearing_precision", "wearing_recall"]
SEG = r"(dev|final|final/source|final/ultralytics)"

for r in RUNS:
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "eval_metrics.py"), r["results"]],
        capture_output=True, text=True, check=True).stdout
    metrics, params, item = {}, {}, None
    for line in out.splitlines():
        m = re.match(r"=== (\w+) ===", line)
        if m:
            item = m.group(1)
            continue
        m = re.match(r"\s+CHOSEN on dev: prompt=(\S+) thr=(\S+)", line)
        if m and item:
            params[f"{item}_prompt"] = m.group(1)
            params[f"{item}_threshold"] = m.group(2)
            continue
        m = re.match(r"person detector \[(\w+)\]: GT=(\d+) pred=(\d+) matched=(\d+) recall=(\S+) precision=(\S+)", line)
        if m:
            metrics[f"person_{m.group(1)}_recall"] = float(m.group(5))
            metrics[f"person_{m.group(1)}_precision"] = float(m.group(6))
            continue
        m = re.match(SEG + r"\s+n=\s*(\d+) unknown=\s*(\d+) \| missing-detect P=(\S+) R\(matched\)=(\S+) R\(end-to-end\)=(\S+) \| wearing P=(\S+) R=(\S+)", line.strip())
        if m and item:
            seg = m.group(1).replace("/", "_")
            metrics[f"{item}_{seg}_n"] = int(m.group(2))
            metrics[f"{item}_{seg}_unknown"] = int(m.group(3))
            for k, v in zip(KEYS, m.groups()[3:]):
                try:
                    metrics[f"{item}_{seg}_{k}"] = float(v)
                except ValueError:
                    pass
    with mlflow.start_run(run_name=f"{AUTHOR}_{r['name']}_{DATE}"):
        mlflow.set_tags({"author": AUTHOR, "git_commit": commit,
                         "dataset_version": DATASET_VERSION, "git_dirty": str(dirty)})
        mlflow.log_params({
            "person_model": "yolo26s.pt", "person_conf": 0.25, "iou_match": 0.5,
            "ppe_model": r["ppe_model"], "ppe_prompts": r["ppe_prompts"],
            "split_rule": "md5(image name) % 10 < 3 -> dev, else final",
            "unknown_policy": "excluded from metrics",
            "ultralytics_version": version("ultralytics"),
            **params})
        mlflow.log_metrics(metrics)
        mlflow.log_text(out, "metrics_report.txt")
        mlflow.log_artifact(os.path.join(r["results"], "persons.csv"))
        mlflow.log_artifact(os.path.join(r["results"], "images.csv"))
    print(f"{r['name']}: logged {len(metrics)} metrics | "
          f"helmet final e2e recall={metrics.get('helmet_final_missing_recall_e2e')} | "
          f"vest final e2e recall={metrics.get('vest_final_missing_recall_e2e')}")
