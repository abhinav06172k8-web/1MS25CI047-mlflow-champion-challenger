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

```bash
# 0) clone and install (Python 3.10+; CPU is enough)
git clone https://github.com/<your-github-username>/1MS25CI047-mlflow-champion-challenger.git
cd 1MS25CI047-mlflow-champion-challenger
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1) terminal 1 - start the MLflow tracking server (Lab 7)
python -m mlflow server --host 127.0.0.1 --port 5000

# 2) terminal 2 - run both models on the same 200 inputs and log everything
python evaluate.py                 # add --samples 10 for a quick debug run first

# 3) register both models as "sentiment-model" and set @champion / @challenger
python register_champion.py

# 4) load the Champion by alias and predict on new data
python predict_from_mlflow.py                    # Champion only
python predict_from_mlflow.py --with-challenger  # Champion and Challenger side by side

# 5) stretch goal - flip the aliases, rerun step 4 without editing it, then flip back
python register_champion.py --swap
python predict_from_mlflow.py
python register_champion.py
```

The tracking URI defaults to `http://127.0.0.1:5000` (`config.py`). To run without a server, set
`MLFLOW_TRACKING_URI=sqlite:///mlflow.db` and open the UI later with `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`.
First runs are slow because both checkpoints (~1.7 GB together) are downloaded once into the Hugging Face cache.

### Repository layout

```
├── README.md
├── requirements.txt
├── config.py                 # all settings + the Champion rule
├── hf_sentiment_model.py     # custom mlflow.pyfunc.PythonModel wrapper (Lab 9)
├── evaluate.py               # runs A and B on the same inputs, logs to MLflow, applies the rule
├── register_champion.py      # registers both models, sets @champion / @challenger
├── predict_from_mlflow.py    # loads models:/sentiment-model@champion, predicts on new data
├── data/
│   ├── eval_inputs.csv       # 200 shared evaluation sentences + reference labels
│   └── new_data.csv          # 8 new sentences written by me (not from SST-2)
├── outputs/                  # predictions, metrics, plots, disagreements, new-data predictions
└── screenshots/              # MLflow compare view, registry aliases, predict_from_mlflow terminal output
```

### What gets logged in MLflow

Experiment **`hf-sentiment-champion-challenger`**, one summary run with two nested model runs:

| Run | Contents |
|---|---|
| `compare_sentiment` (summary) | params `primary_metric`, `margin`, `max_latency_ratio`, `tie_breaker`, `eval_samples`, `seed`; metrics `delta_primary_metric`, `latency_ratio_B_over_A`, both models' macro-F1 and latency; tags `winner`, `winner_hf_model_id`, `decision_reason`, `student_usn`; artifacts: comparison bar chart, latency chart, confidence histogram, `metrics_summary.json`, `disagreements_top5.csv`, `eval_inputs.csv`, `README.md` |
| `A_sentiment_distilbert-sst2` | params `hf_model_id`, `task`, `role=champion`, `batch_size`, `max_length`, `eval_samples`, `seed`, `device`; metrics accuracy, macro-F1, macro precision/recall, F1 per class, `n_errors`, `avg_latency_ms`, `total_inference_s`; tags `student_usn`, `dataset_name`, transformers/torch/mlflow versions; artifacts: predictions CSV + table, metrics JSON, confusion matrix; the pyfunc **model** |
| `B_sentiment_roberta-large` | same as A, with `role=challenger` |

Both models are logged as a custom `mlflow.pyfunc.PythonModel` (`hf_sentiment_model.py`). The Hugging Face ID is stored in the model's `model_config`, so the loaded model knows which checkpoint it is. Each model returns a DataFrame with `text`, `prediction` and `score`, and loads from `runs:/<run_id>/model` or `models:/sentiment-model@champion`.

## 3. Champion rule (written before the final run)

- **Primary metric:** macro-F1 on the 200-sentence evaluation set. **Higher is better.**
- **Minimum improvement margin:** the Challenger must beat the Champion by **at least 1 percentage point** (0.01) of macro-F1, otherwise the Champion stays. With 200 examples one example is 0.5 pp, so a 1 pp margin means the Challenger has to get at least two more examples right, a smaller gap is likely noise.
- **Tie-breaker:** if the difference is within the margin, prefer the **faster** model (lower `avg_latency_ms`).
- **Guardrail:** reject the Challenger outright if its average latency is **more than 10× the Champion's** on the same CPU, since a model that slow is not practical to serve on a laptop or small server.
- **Directions of every metric:** accuracy, macro-F1, precision, recall, F1 per class: higher is better. `avg_latency_ms`, `total_inference_s`, `n_errors`: lower is better.

The same rule is in `config.py` and is logged in the summary run as `primary_metric`, `margin`, `max_latency_ratio` and `tie_breaker`.

## 4. Results table

200 sentences from the SST-2 validation split, seed 42, CPU (2 cores), batch size 8, max length 128.

| Metric (direction) | A: DistilBERT SST-2 (Champion) | B: RoBERTa-large (Challenger) | B − A |
|---|---|---|---|
| **Macro-F1** (↑, primary) | **0.9141** | **0.9344** | **+0.0204 (+2.04 pp)** |
| Accuracy (↑) | 0.9150 | 0.9350 | +0.0200 |
| Macro precision (↑) | 0.9183 | 0.9371 | +0.0188 |
| Macro recall (↑) | 0.9121 | 0.9329 | +0.0208 |
| F1 positive (↑) | 0.9231 | 0.9406 | +0.0175 |
| F1 negative (↑) | 0.9050 | 0.9282 | +0.0232 |
| Errors out of 200 (↓) | 17 | 13 | −4 |
| Avg latency per example, ms (↓) | 35.1 | 222.2 | 6.34× slower |
| Total inference time, s (↓) | 7.01 | 44.44 | +37.4 s |
| Parameters | ~67M | ~355M | ~5.3× |

The two models disagree on 16 of 200 sentences: B is right on 10, A is right on 6 (`outputs/disagreements_all.csv`). Five of them, with my comments, are in `outputs/disagreements_top5.csv`.

Plots (in `outputs/` and logged to MLflow): `comparison_metrics_bar.png`, `comparison_latency_bar.png`, `confidence_distribution.png`, `confusion_matrix_distilbert-sst2.png`, `confusion_matrix_roberta-large.png`.

### Five qualitative disagreements

| id | Sentence (shortened) | Truth | A | B | Comment |
|---|---|---|---|---|---|
| 6 | the film tunes into a grief that could lead a man across centuries | pos | neg | pos | Mood words (grief) fool DistilBERT; RoBERTa reads the tone as appreciative. |
| 18 | true tale of courage... at auschwitz is a harrowing drama... | pos | neg | pos | Dark subject reads as negative to A; B separates topic from verdict. |
| 35 | i'll bet the video game is a lot more fun than the film | neg | pos | neg | Sarcasm with no negative word; only B catches the put-down. |
| 42 | the longer the movie goes, the worse it gets, but it's actually pretty good in the first few minutes | neg | pos | neg | The "but ... pretty good" ending misleads A. |
| 61 | mcconaughey's fun to watch, the dragons are okay, not much fire in the script | pos | pos | neg | Mixed review: A is right with low confidence, B is confidently wrong. |

### Prediction on new data (Step 5.6)

`predict_from_mlflow.py` loads `models:/sentiment-model@champion` → **version 2, `siebert/sentiment-roberta-large-english`** (read from the registry and the model's `model_config`, never hard-coded). On my 8 new sentences (`data/new_data.csv`, checked in code to not overlap the evaluation set) the Champion matched my expected label on 8/8; with `--with-challenger`, DistilBERT got 6/8 (it called "The MLflow UI made comparing my runs surprisingly painless" negative and "Customer support kept me on hold... then hung up" positive). Output: `outputs/new_data_predictions.csv`.

**Stretch goal:** `python register_champion.py --swap` moved `@champion` to version 1; rerunning the *unchanged* `predict_from_mlflow.py` loaded version 1 / DistilBERT instead (`outputs/new_data_predictions_after_alias_swap.csv`). The aliases were then set back to the rule's decision.


### Screenshots

| File | Shows |
|---|---|
| `screenshots/01_mlflow_compare_view.png` | MLflow Compare view of `A_sentiment_distilbert-sst2` vs `B_sentiment_roberta-large` |
| `screenshots/02_registry_aliases.png` | `sentiment-model` in the registry: v2 `@champion`, v1 `@challenger` |
| `screenshots/03_predict_from_mlflow_terminal.png` | Terminal output of `python predict_from_mlflow.py --with-challenger` |
| `screenshots/00_mlflow_experiment_runs.png` | The experiment's run list (summary run + two nested runs) |

## 5. Decision

**Champion: Model B, `siebert/sentiment-roberta-large-english` (registry version 2, alias `@champion`). DistilBERT keeps `@challenger`.**

Following the rule fixed in section 3, the primary metric is macro-F1. RoBERTa-large scored 0.9344 against DistilBERT's 0.9141, an improvement of 2.04 percentage points, which is above the 1 pp margin. The latency guardrail also passes: RoBERTa-large averaged 222 ms per sentence against 35 ms, a 6.3× slowdown, inside the 10× limit. The tie-breaker was not needed. So by the rule the Challenger replaces the Champion, and the gain is consistent across every quality metric, both classes improve, and the errors drop from 17 to 13.

Is the gain meaningful? Only partly. With 200 sentences one example is 0.5 pp, so the gap is four sentences. The models disagree on 16 sentences and B wins 10 of them; an exact McNemar (sign) test on 10 vs 6 gives p ≈ 0.45, so this sample cannot prove the difference is not noise. The qualitative cases point the same way as the numbers, though: B handles sarcasm, contrast ("but") and sad topics in positive reviews better, which matches its larger size and broader training data. Note also that DistilBERT was fine-tuned on the SST-2 training split, so this test set is in-domain for A and, if anything, favours the old Champion.

The trade-off is cost: B is ~5× larger and ~6× slower on CPU. For an offline batch job that is acceptable; for a high-traffic, low-latency API the old Champion would be the safer choice, and a larger evaluation set should be run before such a switch.

## 6. Limitations and declaration

- **Sample size:** 200 sentences from one dataset (movie reviews). A 2 pp gap is 4 sentences and is not statistically significant (McNemar p ≈ 0.45). Results may not transfer to tweets, product reviews or Indian-English text.
- **Domain bias:** DistilBERT was fine-tuned on SST-2 train, so this test is in-domain for Model A; RoBERTa-large (SiEBERT) was trained on 15 mixed datasets.
- **Latency** was measured on a 2-core CPU with batch size 8 and truncation at 128 tokens, after one warm-up call; numbers will differ on other hardware and on GPU.
- **New data** is only 8 short sentences written by me, so it shows the pipeline works rather than measuring accuracy.
- The logged pyfunc model stores the Hugging Face model ID and downloads the weights from the Hub in `load_context` (same approach as Lab 9), so the weights are not stored in MLflow; loading needs internet the first time.
- `mlruns/`, `mlartifacts/`, `mlflow.db` and model files are excluded by `.gitignore` as required.
- **AI assistance declaration:** I used an AI assistant (Hark) to help write and debug the code and to draft parts of this README. I have read and run every script, and I can explain each line and design choice in the viva.
