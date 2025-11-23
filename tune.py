import subprocess
import json
import os
import sys
import time
import shutil

# Paths
TRAIN_PATH = "data/train.jsonl"
DEV_PATH = "data/dev.jsonl"
MODEL_OUT = "out_tune"

# Hyperparameter search space
LEARNING_RATES = [5e-5, 1e-5]
BATCH_SIZES = [16]
EPOCHS = [4]
TOKEN_THRESHOLDS = [0.55, 0.60]

RESULTS_FILE = "tuning_results.json"


def run_cmd(cmd_list):
    """Run a subprocess and return combined stdout+stderr."""
    print("\n🔹 Running:", " ".join(cmd_list))
    proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()

    if out.strip():
        print(out)
    if err.strip():
        print("⚠ stderr:", err)

    return out + "\n" + err


def parse_eval_output(text):
    """Extract Macro-F1 and PII metrics from eval output."""
    macro_f1 = None
    pii_p = None
    pii_r = None
    pii_f1 = None

    for line in text.splitlines():
        if line.startswith("Macro-F1"):
            macro_f1 = float(line.split(":")[1].strip())

        if line.startswith("PII-only metrics:"):
            parts = line.split()
            for p in parts:
                if p.startswith("P="): pii_p = float(p.split("=")[1])
                if p.startswith("R="): pii_r = float(p.split("=")[1])
                if p.startswith("F1="): pii_f1 = float(p.split("=")[1])

    return macro_f1, pii_p, pii_r, pii_f1


def train_model(lr, batch, epochs):
    """Train model with given hyperparameters."""
    print(f"\n🧹 Cleaning old model directory: {MODEL_OUT}")
    if os.path.exists(MODEL_OUT):
        shutil.rmtree(MODEL_OUT)

    cmd = [
        sys.executable, "src/train.py",
        "--model_name", "distilbert-base-uncased",
        "--train", TRAIN_PATH,
        "--dev", DEV_PATH,
        "--out_dir", MODEL_OUT,
        "--batch_size", str(batch),
        "--epochs", str(epochs),
        "--lr", str(lr)
    ]
    run_cmd(cmd)


def run_predict(threshold):
    """Run prediction for given threshold."""
    cmd = [
        sys.executable, "src/predict.py",
        "--model_dir", MODEL_OUT,
        "--input", DEV_PATH,
        "--output", "dev_pred.json",
        "--token_threshold", str(threshold),
        "--device", "cpu"
    ]
    run_cmd(cmd)


def run_eval():
    """Evaluate predictions."""
    cmd = [
        sys.executable, "src/eval_span_f1.py",
        "--gold", DEV_PATH,
        "--pred", "dev_pred.json"
    ]
    return run_cmd(cmd)


def main():
    print("\n🚀 Starting Hyperparameter Tuning...\n")

    results = []
    total_runs = len(LEARNING_RATES) * len(BATCH_SIZES) * len(EPOCHS) * len(TOKEN_THRESHOLDS)
    run_index = 1

    for lr in LEARNING_RATES:
        for batch in BATCH_SIZES:
            for ep in EPOCHS:

                print("\n========================================")
                print(f" TRAINING CONFIG → LR={lr}, Batch={batch}, Epochs={ep}")
                print("========================================")

                train_model(lr, batch, ep)

                for th in TOKEN_THRESHOLDS:
                    print(f"\n🔸 Running prediction with threshold={th}")

                    run_predict(th)
                    eval_output = run_eval()

                    macro_f1, pii_p, pii_r, pii_f1 = parse_eval_output(eval_output)

                    result = {
                        "lr": lr,
                        "batch": batch,
                        "epochs": ep,
                        "threshold": th,
                        "macro_f1": macro_f1,
                        "pii_precision": pii_p,
                        "pii_recall": pii_r,
                        "pii_f1": pii_f1
                    }

                    results.append(result)

                    print(f"\n✔ RESULT #{run_index}/{total_runs}:")
                    print(json.dumps(result, indent=2))

                    run_index += 1

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n📁 All tuning results saved to:", RESULTS_FILE)

    # Select best by PII precision then Macro-F1
    best = sorted(
        results,
        key=lambda x: (x["pii_precision"], x["macro_f1"]),
        reverse=True
    )[0]

    print("\n🏆 BEST CONFIGURATION FOUND:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
