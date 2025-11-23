# threshold_grid.py
import subprocess, json, argparse, os, sys, tempfile

def run_predict_and_eval(model_dir, dev_path, out_json, token_threshold):
    cmd = [
        sys.executable, "src/predict.py",
        "--model_dir", model_dir,
        "--input", dev_path,
        "--output", out_json,
        "--token_threshold", str(token_threshold),
        "--device", "cpu"
    ]
    subprocess.check_call(cmd)
    ev_cmd = [sys.executable, "src/eval_span_f1.py", "--gold", dev_path, "--pred", out_json]
    proc = subprocess.Popen(ev_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5,0.6,0.65,0.7,0.75,0.8])
    args = ap.parse_args()
    print("Thresholds:", args.thresholds)
    results = {}
    tmp_out = "out/dev_pred_grid.json"
    for t in args.thresholds:
        print("Running threshold", t)
        out = run_predict_and_eval(args.model_dir, args.dev, tmp_out, t)
        print(out)
        piip = None
        for line in out.splitlines():
            if line.startswith("PII-only metrics:"):
                parts = line.split()
                # expects: PII-only metrics: P=0.800 R=0.600 F1=0.686
                for p in parts:
                    if p.startswith("P="):
                        try:
                            piip = float(p.split("=")[1])
                        except:
                            pass
        results[t] = piip
    print("Summary PII precision by threshold:")
    for t, p in results.items():
        print(t, p)

if __name__ == "__main__":
    main()
