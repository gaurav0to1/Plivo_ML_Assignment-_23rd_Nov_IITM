# quantize_and_measure.py
import torch
import json, argparse, statistics, time
from transformers import AutoTokenizer, AutoModelForTokenClassification

def quantize_model(model):
    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return q_model

def measure(model, tokenizer, texts, runs=50, max_length=256, device="cpu"):
    model.to(device)
    model.eval()
    times = []
    # warmup
    with torch.no_grad():
        for _ in range(5):
            enc = tokenizer(texts[0], truncation=True, max_length=max_length, return_tensors="pt")
            _ = model(**{k: v.to(device) for k, v in enc.items()})
    for i in range(runs):
        t = texts[i % len(texts)]
        enc = tokenizer(t, truncation=True, max_length=max_length, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**{k: v.to(device) for k, v in enc.items()})
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    p50 = statistics.median(times)
    times_sorted = sorted(times)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]
    return p50, p95

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    qmodel = quantize_model(model)
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    p50, p95 = measure(qmodel, tokenizer, texts, runs=args.runs, device="cpu")
    print(f"Quantized model latency over {args.runs} runs: p50={p50:.2f} ms p95={p95:.2f} ms")

if __name__ == "__main__":
    main()
