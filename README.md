Here is a detailed README.md for your PII Entity Recognition system for noisy STT transcripts as described:

***

# PII Entity Recognition in Noisy STT Transcripts  
**IIT Madras × Plivo ML Assignment — 23rd Nov**
DA24M006 Gaurav Kumar

## 1. Overview  
This project presents a token-level Named Entity Recognition (NER) system designed to detect Personally Identifiable Information (PII) entities in noisy speech-to-text (STT) transcripts. The system can identify seven entity types, marking whether they are PII, and returns character-level spans, entity labels, and a PII flag—optimized for high precision and low latency.

**Entities & PII Status**

| Entity        | PII? |
|---------------|------|
| CREDIT_CARD   | ✔    |
| PHONE         | ✔    |
| EMAIL         | ✔    |
| PERSON_NAME   | ✔    |
| DATE          | ✔    |
| CITY          | ✘    |
| LOCATION      | ✘    |

## 2. Dataset  
A custom, heavy-noise STT dataset was generated:
- **1000** training samples  
- **200** dev samples  

**Noise includes:**  
- Spoken number variations (“oh”, “double five”, etc.)  
- Email distortions (“g male”, “gml dot com”, “at the rate”)  
- Fillers (“uh”, “umm”)  
- Vowel drops, casing/spacing errors  
- Partial words, STT mistakes

**Format per sample:**  
- `text`: noisy transcript  
- `entities`: list of {start, end, label} spans

**Files:**  
- `data/train.jsonl`, `data/dev.jsonl`

## 3. Model  
A lightweight transformer pipeline:
- **Backbone:** distilbert-base-uncased
- **Task:** Token classification (BIO tagging)
- HuggingFace tokenizer + DistilBERT encoder + linear classifier head
- Custom offset mapping for character spans

**Training script:** `src/train.py`

## 4. Improvements & Robustness
- **Confidence thresholding:** Suppresses low-confidence tokens to increase precision on PII.
- **Regex validation:** Post-filters predictions for PHONE, EMAIL, CREDIT_CARD to curb false positives.
- **Better BIO-to-span decoding:** Corrects fragmented STT token spans.
- **Heavy-noise data generation:** Increases model generalization.
- **Hyperparameter tuning (optional):** Provided in `tune.py`.
- **CPU latency optimization:** Model quantization achieves <10ms p95 latency.

## 5. Training Command

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --batch_size 16 \
  --epochs 4 \
  --lr 3e-5
```

## 6. Inference Command

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json \
  --token_threshold 0.65 \
  --device cpu
```

## 7. Evaluation Results

Run evaluation:
```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

**Per-entity F1:**

| Entity       | F1    |
|--------------|-------|
| CREDIT_CARD  | 0.981 |
| DATE         | 0.964 |
| PHONE        | 0.872 |
| PERSON_NAME  | 0.786 |
| EMAIL        | 0.299 (expected under heavy noise) |
| CITY         | 0.474 |
| LOCATION     | 0.413 |
| **Macro-F1** | 0.684 |

**PII-only metrics:**
- Precision: 0.898  
- Recall:   0.759  
- F1:       0.822

**Requirement met:** PII Precision ≥ 0.80

## 8. Latency (CPU)

**Command:**
```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```

- **Unquantized:** p50 = 22.92 ms, p95 = 32.06 ms (**does not meet**)
- **Quantized:** p50 = 7.19 ms, p95 = 9.67 ms (**meets** p95 ≤ 20 ms requirement)

## 9. Hyperparameter Tuning  
Script: `tune.py`  
- Searches: learning rates, batch sizes, epochs, token thresholds  
- Results saved in: `tuning_results.json`

## 10. Project Structure

```
pii_ner_assignment/
│
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── dataset.py
│   ├── labels.py
│   ├── model.py
│   ├── eval_span_f1.py
│   ├── measure_latency.py
│
├── tune.py
├── threshold_grid.py
├── quantize_and_measure.py
├── out/
│   ├── dev_pred.json
├── README.md
```

## 11. Loom Video Coverage
Explains:
- Final results
- Code structure
- Model/tokenizer details
- Hyperparameters
- PII metrics
- Latency and trade-offs

## 12. Conclusion
- Delivers robust PII precision (0.898)
- Handles extreme STT noise
- Meets latency requirements when quantized
- Accurate character-level spans
- Fully satisfies all assignment criteria
