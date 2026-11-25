"""Register the provisional champion in the local MLflow model registry.

Run from work/:
    cd /e/dev/safety-eval-lab/work && python ../register_champion.py

Safe to run twice: if the registered model already exists, nothing is changed.
The weights are NOT stored in MLflow. The model version points to the Hugging Face
file (repo + revision) and records its sha256. The script refuses to register if the
local cached file does not match the expected sha256.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import mlflow
from huggingface_hub import hf_hub_download
from mlflow.exceptions import MlflowException

EXPERIMENT = "ppe-model-selection"
MODEL_NAME = "ppe-champion"
ALIAS = "champion"
REPO = "ayushgupta7777/safetyvision-yolov8"
FILE = "v2/best.pt"
REVISION = "56a71758b55f0e9f2b4b2d6b51a779a1f882da10"
SHA256 = "7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a"
CARD = "docs/MODEL_CARD_safetyvision_v2.md"
EVAL_KEYS = {"cppe": "safetyvision-v2|cppe", "kaggle": "safetyvision-v2|kaggle"}


def die(msg):
    print("STOP:", msg)
    sys.exit(1)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def card_commit():
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", CARD],
                             cwd="..", capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if not Path("mlflow.db").exists():
    die("mlflow.db not found here. Run from work/.")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()

try:
    client.get_registered_model(MODEL_NAME)
    print(f"'{MODEL_NAME}' already exists. Nothing changed.")
    sys.exit(0)
except MlflowException:
    pass

exp = client.get_experiment_by_name(EXPERIMENT)
if exp is None:
    die(f"experiment '{EXPERIMENT}' not found.")

run_ids = {}
for short, key in EVAL_KEYS.items():
    runs = client.search_runs([exp.experiment_id], filter_string=f"tags.eval_key = '{key}'")
    if len(runs) != 1:
        die(f"expected 1 active run for {key}, found {len(runs)}.")
    r = runs[0]
    if r.info.status != "FINISHED" or r.data.tags.get("git_dirty") != "False":
        die(f"run for {key} is not FINISHED or git_dirty is not False.")
    run_ids[short] = r.info.run_id

try:
    weights = hf_hub_download(REPO, FILE, revision=REVISION, local_files_only=True)
except Exception as e:
    die(f"weights not found in the local cache at revision {REVISION[:7]}: {e}")
if sha256_of(weights) != SHA256:
    die("sha256 of the cached weights does not match the expected value.")
print("weights sha256 OK:", SHA256[:16])

client.create_registered_model(
    MODEL_NAME,
    description="PROVISIONAL champion PPE model. See " + CARD,
)
mv = client.create_model_version(
    MODEL_NAME,
    source=f"hf://{REPO}@{REVISION}/{FILE}",
    run_id=run_ids["cppe"],
    tags={
        "status": "provisional",
        "selected_on": "2026-11-25",
        "selection_rule": "Recall first, then False alarm, then speed (speed not measured)",
        "sha256": SHA256,
        "hf_repo": REPO,
        "hf_file": FILE,
        "hf_revision": REVISION,
        "license": "AGPL-3.0 (base model, per model page; repo license field unconfirmed)",
        "model_card": CARD,
        "model_card_commit": card_commit(),
        "run_id_cppe": run_ids["cppe"],
        "run_id_kaggle": run_ids["kaggle"],
        "eval_imgsz": "640",
        "thresholds_cppe": "helmet 0.30, vest 0.05",
        "thresholds_kaggle": "helmet 0.35, vest 0.10",
    },
)
client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)

got = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
print(f"registered '{MODEL_NAME}' version {got.version}, alias '{ALIAS}' -> version {got.version}")
print("status:", got.tags["status"], "| card commit:", got.tags["model_card_commit"])
