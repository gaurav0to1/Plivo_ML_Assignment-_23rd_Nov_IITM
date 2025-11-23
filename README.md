PII Entity Recognition in Noisy STT Transcripts
IIT Madras × Plivo ML Assignment — 23rd Nov
DA24M006 GAURAV KUMAR
1. Overview

This project builds a token-level NER system to detect PII entities from noisy Speech-to-Text (STT) transcripts.
The model identifies 7 entities and marks whether they are PII:

Entity	PII?
CREDIT_CARD	✔
PHONE	✔
EMAIL	✔
PERSON_NAME	✔
DATE	✔
CITY	✘
LOCATION	✘

The system outputs character-level spans, entity labels, and a PII flag, with high precision and low latency.

2. Dataset

As required, I generated a custom heavy-noise STT-style dataset:

1000 training examples

200 dev examples

Noise includes:

spoken numbers (“oh”, “double five”, “four two”)

email distortions (“g male”, “gml dot com”, “at the rate”)

filler words (“uh”, “umm”, “you know”)

vowel drops, casing noise, spacing errors

partial word segments and STT-like mistakes

Files:

data/train.jsonl
data/dev.jsonl


Each sample contains:

text: noisy transcript

entities: list of {start, end, label} spans

3. Model

A lightweight transformer:

distilbert-base-uncased


trained as a token classification model (BIO tagging).

Key components:

HuggingFace tokenizer

DistilBERT encoder

Linear classification head

Offset mapping → character spans

Training located in:

src/train.py

4. Improvements

To boost precision and robustness on heavy-STT noise:

✔ Confidence thresholding

Suppress low-confidence tokens → higher PII precision.

✔ Regex validation

Applied to PHONE, EMAIL, CREDIT_CARD to reduce false positives.

✔ Improved BIO-to-span decoding

Handles fragmented STT tokens correctly.

✔ Heavy-noise data generation

Ensures the model generalizes beyond clean text.

✔ Optional hyperparameter tuning

Provided in tune.py.

✔ CPU latency optimization (quantization)

Quantized model runs under 10 ms p95.

5. Training Command
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --batch_size 16 \
  --epochs 4 \
  --lr 3e-5

6. Inference Command
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json \
  --token_threshold 0.65 \
  --device cpu

7. Evaluation Results

Using:

python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json

Per-entity F1

CREDIT_CARD: 0.981

DATE: 0.964

PHONE: 0.872

PERSON_NAME: 0.786

EMAIL: 0.299 (expected under heavy noise)

CITY: 0.474

LOCATION: 0.413

Macro-F1:
0.684

PII-only metrics:
Precision = 0.898
Recall    = 0.759
F1        = 0.822


✔ Meets requirement: PII Precision ≥ 0.80

8. Latency (CPU)

Command:

python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50

Unquantized Model

p50 = 22.92 ms

p95 = 32.06 ms

(Does NOT meet requirement)

Quantized Model

p50 = 7.19 ms

p95 = 9.67 ms

✔ Meets assignment requirement: p95 ≤ 20 ms

9. Hyperparameter Tuning (Optional)

Provided script:

tune.py


Searches:

learning rates

batch sizes

epochs

token thresholds

Results saved in:

tuning_results.json

10. Project Structure
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

11. Loom Video Coverage

In the video, I explain:

Final results

Code structure

Model + tokenizer

Key hyperparameters

PII precision/recall/F1

Latency results + trade-offs

12. Conclusion

This system:

Achieves high PII precision (0.898)

Handles heavy STT noise

Passes latency requirement with quantization

Produces accurate character-level spans

Fully satisfies the assignment requirements
