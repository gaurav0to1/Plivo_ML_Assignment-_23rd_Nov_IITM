import json
import os
import re
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import ID2LABEL, LABEL2ID, label_is_pii

# ----------------------
# STT normalization helpers
# ----------------------
NUM_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10"
}

def normalize_spoken_digits(text: str) -> str:
    # Replace common spoken number words with digits (best-effort).
    # Keep original text too — offsets are computed on original, so only used for regex checks.
    toks = text.split()
    out = []
    for t in toks:
        tl = t.lower().strip(".,")
        if tl in NUM_WORDS:
            out.append(NUM_WORDS[tl])
        else:
            out.append(t)
    return " ".join(out)

def stt_simplify_for_regex(text: str) -> str:
    # Helpful normalization for regex checks: convert "at" and "dot" patterns to symbols
    t = text.lower()
    t = re.sub(r"\s+at\s+", "@", t)
    t = re.sub(r"\s+dot\s+", ".", t)
    t = re.sub(r"\s+dash\s+", "-", t)
    # merge spelled digits to digits for regex: "four two four two" -> "4242"
    parts = t.split()
    merged = []
    i = 0
    while i < len(parts):
        if parts[i] in NUM_WORDS:
            j = i
            digits = []
            while j < len(parts) and parts[j] in NUM_WORDS:
                digits.append(NUM_WORDS[parts[j]])
                j += 1
            if j > i:
                merged.append("".join(digits))
                i = j
                continue
        merged.append(parts[i])
        i += 1
    return " ".join(merged)

# ----------------------
# Regex confirmers
# ----------------------
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-\s]?)?(?:\d[-\s]*){7,15}$")
CREDIT_RE = re.compile(r"^(?:\d[ -]*){12,19}$")
EMAIL_RE = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")

def confirm_span_label(text: str, s: int, e: int, lab: str) -> bool:
    # Use normalized substring for regex tests
    substr = text[s:e].strip()
    if not substr:
        return False
    norm = stt_simplify_for_regex(substr)
    if lab == "PHONE":
        # remove spaces/dashes before test
        cand = re.sub(r"[^\d\+]", "", norm)
        return bool(PHONE_RE.search(cand))
    if lab == "CREDIT_CARD":
        cand = re.sub(r"[^\d]", "", norm)
        return bool(CREDIT_RE.search(cand))
    if lab == "EMAIL":
        return bool(EMAIL_RE.search(norm))
    return True

# ----------------------
# BIO -> spans (conservative)
# ----------------------
def bio_to_spans_from_ids(offsets: List[Tuple[int,int]], label_ids: List[int]) -> List[Tuple[int,int,str]]:
    spans = []
    cur_label = None
    cur_start = None
    cur_end = None
    for (start, end), lid in zip(offsets, label_ids):
        # skip special tokens (0,0 usually)
        if start == end:
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_label = None
            continue
        lab = ID2LABEL.get(int(lid), "O")
        if lab == "O":
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_label = None
            continue
        prefix, ent = lab.split("-", 1)
        if prefix == "B":
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
            cur_label = ent
            cur_start = start
            cur_end = end
        elif prefix == "I":
            if cur_label == ent:
                cur_end = end
            else:
                # stray I- without matching B-: conservatively start new span
                if cur_label is not None:
                    spans.append((cur_start, cur_end, cur_label))
                cur_label = ent
                cur_start = start
                cur_end = end
    if cur_label is not None:
        spans.append((cur_start, cur_end, cur_label))
    return spans

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--token_threshold", type=float, default=0.65, help="Min token prob to accept non-O label")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}
    with open(args.input, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    for obj in lines:
        uid = obj["id"]
        text = obj["text"]
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        offsets = enc["offset_mapping"][0].tolist()
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0]  # (seq_len, num_labels)
            probs = F.softmax(logits, dim=-1)
            max_probs, pred_label_ids = torch.max(probs, dim=-1)
            pred_ids = []
            for lid, mp in zip(pred_label_ids.cpu().tolist(), max_probs.cpu().tolist()):
                lab = ID2LABEL.get(int(lid), "O")
                if lab == "O":
                    pred_ids.append(int(lid))
                else:
                    if mp < args.token_threshold:
                        pred_ids.append(LABEL2ID["O"])
                    else:
                        pred_ids.append(int(lid))

        spans = bio_to_spans_from_ids(offsets, pred_ids)

        # post-filter spans using regex confirmations for high-precision classes
        filtered = []
        for s, e, lab in spans:
            ok = True
            if lab in {"PHONE", "CREDIT_CARD", "EMAIL"}:
                ok = confirm_span_label(text, s, e, lab)
            if ok:
                filtered.append((s, e, lab))

        ents = []
        for s, e, lab in filtered:
            ents.append({"start": int(s), "end": int(e), "label": lab, "pii": bool(label_is_pii(lab))})
        results[uid] = ents

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")

if __name__ == "__main__":
    main()
