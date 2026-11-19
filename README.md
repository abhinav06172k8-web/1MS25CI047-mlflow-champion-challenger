# 1MS25CI047 – MLflow Champion–Challenger (Hugging Face Sentiment)

**Student:** GOLLA ABHINAV · **USN:** 1MS25CI047 · **Section:** A
Skill Lab (MLOps with MLflow) – Mini Project, extension of Labs 4, 5, 8, 9, 10, 11 and 13.

## 1. Task and models

| | |
|---|---|
| Task | Binary sentiment classification (positive / negative) |
| Dataset | GLUE **SST-2**, validation split (held out from training) |
| Subset | 200 sentences, shuffled with seed 42, saved in `data/eval_inputs.csv` |
| Model A (Champion) | `distilbert-base-uncased-finetuned-sst-2-english` (~67M) |
| Model B (Challenger) | `siebert/sentiment-roberta-large-english` (~355M) |

## 2. How to run

## 3. Champion rule (written before the final run)

- **Primary metric:** macro-F1 on the 200-sentence evaluation set. **Higher is better.**
- **Minimum improvement margin:** the Challenger must beat the Champion by **at least 1 percentage point** (0.01) of macro-F1, otherwise the Champion stays. With 200 examples one example is 0.5 pp, so a 1 pp margin means the Challenger has to get at least two more examples right, a smaller gap is likely noise.
- **Tie-breaker:** if the difference is within the margin, prefer the **faster** model (lower `avg_latency_ms`).
- **Guardrail:** reject the Challenger outright if its average latency is **more than 10× the Champion's** on the same CPU, since a model that slow is not practical to serve on a laptop or small server.
- **Directions of every metric:** accuracy, macro-F1, precision, recall, F1 per class: higher is better. `avg_latency_ms`, `total_inference_s`, `n_errors`: lower is better.

The same rule is in `config.py` and is logged in the summary run as `primary_metric`, `margin`, `max_latency_ratio` and `tie_breaker`.

## 4. Results table

## 5. Decision

## 6. Limitations and declaration
