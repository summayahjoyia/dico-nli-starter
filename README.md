# DiCo-NLI Starter Assignment 

SemEval-2027 Task 2, Track 1 (English). All results are on the official dev split (660 items), scored with the official scorer. Train data is the official train split only. Dev data is never used for training.

## AI Assistance

I used Claude (Anthropic) throughout this assignment as a learning and
debugging aid. Specifically:

**Understanding the pilot paper (Apaolaza et al.)**

**Setup and workflow**
Claude walked me through setting up Colab (GPU runtime, GitHub token via
Colab Secrets), connecting to my own GitHub repo separately from the
organizers' task repo, and a git workflow (add/commit/pull --rebase/push).

**Part 2 (DistilBERT fine-tuning)**
I ran the organizers' starter kit (`main.py`) largely as provided. When two
DeBERTa-v3-base runs collapsed to predicting a single label (loss stuck at
~1.39, equal to random guessing among 4 labels), Claude helped me diagnose
this by comparing loss values against the "guessing evenly" benchmark,
and suggested testing DistilBERT as a smaller model to isolate whether the
failure was DeBERTa-specific or a pipeline/data issue.

**Part 3 (prompted LLM)**
I chose Qwen2.5-1.5B-Instruct (Claude explained the tradeoffs vs.
Llama-3.2-3B-Instruct — mainly that Qwen is ungated and smaller, so faster
to set up and run on a free Colab GPU). 

**Organizing this README**
Using my Colab notebook as a reference, I asked Claude to help
structure this README's reproduction steps, results tables, and
hyperparameter listings from the code.

## Setup
- Platform: Google Colab, T4 GPU
- Task repo (data, scorer, starter kit): https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI
- Starter kit installed with `pip install -r requirements.txt` from `starter_kit/`

## How to reproduce


## Setup (run once per Colab session, in order)

1. Add a Colab Secret named `GITHUB_TOKEN` (fine-grained PAT scoped to this repo only).
2. Runtime > Change runtime type > **T4 GPU**.
3. Run the notebook's setup cells (clone this repo, clone the task repo, load
   `train`/`dev` with pandas) before running any part below.

---

## Part 0: Trivial baseline

Predicts the most frequent train label for every dev row, no model involved.

```python
majority = train["label"].value_counts().idxmax()   # "FORWARD_ENTAILMENT"
pred = pd.DataFrame({"instance_id": dev["instance_id"], "label": majority})
```

Scored with:
```bash
python3 -m evaluation_functions \
    --gold final_data/dev/dico_nli_dev_track1_reference.csv \
    --predictions results/majority_dev.csv \
    --output-dir results/majority_scores
```

**Results:** Weighted F1 0.1287, SoftCons 0.0, HardCons 0.0.

---

## Part 1: Data exploration

- `results/part1/label_distribution.csv` — label counts/percentages, train vs dev.
- `results/part1/examples.csv` — 3 hand-annotated examples per label with reasoning.
- **Reversed-pair structure:** each pair is stored as two rows sharing one `pair_id`,
  with `instance_id` ending in `__original` or `__flipped`. In the dev reference file,
  `reverse_pair_id` holds the twin's `instance_id`. The 106 dev rows with no
  `reverse_pair_id` are exactly the rows labeled `NEGATIVE_OTHER`.

---

## Part 2: System A — Fine-tuned encoder

### Failed attempts: DeBERTa-v3-base (seed 42)

Two runs, differing only in learning rate, both collapsed to predicting a single
label across the entire dev set in every epoch:

```bash
cd /content/task_repo/starter_kit
python -u main.py \
    --model_path "microsoft/deberta-v3-base" \
    --tokenizer_path "microsoft/deberta-v3-base" \
    --do_train \
    --train_path "../final_data/train/dico_nli_train_track1_participant_labeled.csv" \
    --do_dev \
    --dev_path "../final_data/dev/dico_nli_dev_track1_participant_labeled.csv" \
    --reference_path "../final_data/dev/dico_nli_dev_track1_reference.csv" \
    --no-do_test \
    --seed 42 --dropout 0.1 --warmup_pcrt 0.01 --lr 5e-5 \
    --batch_size 32 --wd 1e-5 --num_epochs 3 \
    --no-is_optuna_trial --max_length 256 \
    --results_dir "results/deberta_seed42"
```

Rerun with `--lr 2e-5` (results_dir `deberta_seed42_lr2e-5`), same collapse.
Training loss settled at ~1.39, which is equal to random guessing among 4 labels. See
`results/deberta_seed42_failed_lr5e-5/` and `results/deberta_seed42_failed_lr2e-5/`.

### System A: DistilBERT (distilbert-base-uncased), seeds 42 and 43

Same command, swapping the model and running two seeds:

```bash
cd /content/task_repo/starter_kit
python -u main.py \
    --model_path "distilbert-base-uncased" \
    --tokenizer_path "distilbert-base-uncased" \
    --do_train \
    --train_path "../final_data/train/dico_nli_train_track1_participant_labeled.csv" \
    --do_dev \
    --dev_path "../final_data/dev/dico_nli_dev_track1_participant_labeled.csv" \
    --reference_path "../final_data/dev/dico_nli_dev_track1_reference.csv" \
    --no-do_test \
    --seed 42 --dropout 0.1 --warmup_pcrt 0.01 --lr 2e-5 \
    --batch_size 32 --wd 1e-5 --num_epochs 3 \
    --no-is_optuna_trial --max_length 256 \
    --results_dir "results/distilbert_seed42"
```

Repeat with `--seed 43 --results_dir "results/distilbert_seed43"` for the second seed.

**Hyperparameters:** lr=2e-5, batch_size=32, weight_decay=1e-5, dropout=0.1 (ignored
— DistilBERT's config has no `classifier_dropout` attribute, so this flag has no
effect and the model uses its built-in default), warmup_pcrt=0.01, num_epochs=3,
max_length=256.

The kit scores dev automatically after every epoch. **We report the final (3rd)
epoch only**, a rule fixed in advance rather than chosen by comparing dev scores
across epochs, to avoid using dev to select the model.

### Results (final epoch)

| Metric | Seed 42 | Seed 43 | Mean | Std (n=2) |
|---|---|---|---|---|
| Weighted F1 | 0.4090 | 0.4607 | 0.4349 | 0.0366 |
| SoftCons | 0.3574 | 0.3935 | 0.3755 | 0.0255 |
| HardCons | 0.2491 | 0.2924 | 0.2708 | 0.0306 |

Std = sample standard deviation across the 2 seeds; with only 2 runs this is a
rough estimate of spread, not a robust one.

Clean `instance_id,label` predictions, re-scored as a sanity check (numbers match
the kit's own epoch-3 output exactly):
`predictions/distilbert_seed42_dev.csv`, `predictions/distilbert_seed43_dev.csv`
`results/distilbert_seed42_scores/`, `results/distilbert_seed43_scores/`
`results/distilbert_per_seed_scores.csv`, `results/distilbert_mean_std_range.csv`

---

## Part 3: System B — Prompted LLM

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`, loaded in float16 (`dtype=torch.float16`),
`device_map="auto"`. Not gated (unlike Llama-3.2-3B-Instruct), so no HF token or
access approval needed (chosen for faster setup and inference on a free Colab GPU.)

**Generation settings:** `max_new_tokens=10`, `do_sample=False` (deterministic,
reproducible output, always picks the model's most likely next token).

### Zero-shot prompt
Compare text1 and text2 and assess their relationship, after assessing it, label
them from this set of labels, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT,
EQUIVALENCE, NEGATIVE_OTHER.

Explanations of each label:

FORWARD_ENTAILMENT: text1 is the more specific one, and if text1 is true,
text2 must also be true, but not the other way around.
BACKWARD_ENTAILMENT: text2 is the more specific one, and if text2 is true,
text1 must also be true, but not the other way around.
EQUIVALENCE: true in both directions, both phrases are equivalent.
NEGATIVE_OTHER: none of the other 3 labels match this pair.

Respond only with the label, do not explain your answer
text1: {text1} text2: {text2} Label:


### Few-shot prompt

Same instructions as zero-shot, with 4 worked examples (one per label, from
`train`, never `dev`) inserted before the real question:

| text1 | text2 | label |
|---|---|---|
| A woman and child | A women and a young girl | BACKWARD_ENTAILMENT |
| All red-painted sports automobile | Every red sports car | EQUIVALENCE |
| in Egypt | Egypt | FORWARD_ENTAILMENT |
| militants | as Sinai attack | NEGATIVE_OTHER |

> **Known issue:** the few-shot prompt's BACKWARD_ENTAILMENT definition contained
> a typo ("text2 is theok l more specific one") left over from drafting. It was
> present during the actual run that produced the reported few-shot scores below.
> Given the strong few-shot results, it does not appear to have meaningfully
> harmed the model's understanding, but it is noted here for transparency. It has been corrected in the latest notebook in the repository but current results were produced with the typo included.

### Parser

```python
valid_labels = ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "EQUIVALENCE", "NEGATIVE_OTHER"]

def parse_label(raw_output):
    cleaned = raw_output.strip().upper()
    for label in valid_labels:
        if label in cleaned:
            return label
    return None
```

Checks whether any of the 4 real label strings appears anywhere in the model's
output (case-insensitive, whitespace-trimmed). Unparseable outputs fall back to
`FORWARD_ENTAILMENT` (the majority label from Part 0) and are logged with their
raw text for reporting.

**Unparseable outputs: 0/660 for zero-shot, 0/660 for few-shot.** The fallback
was never actually triggered.

### Results

| Metric | Zero-shot | Few-shot |
|---|---|---|
| Weighted F1 | 0.2025 | 0.4194 |
| SoftCons | 0.2130 | 0.3357 |
| HardCons | 0.0325 | 0.2238 |
| Accuracy | 0.2485 | 0.4242 |
| Macro F1 | 0.2057 | 0.4148 |

Few-shot more than doubled zero-shot's weighted F1. The biggest driver: zero-shot
predicted EQUIVALENCE only 3/660 times (F1 0.023); few-shot predicted it 245/660
times (F1 0.496), at some cost of over-predicting EQUIVALENCE on genuine
FORWARD/BACKWARD pairs (see confusion matrices in the notebook, cells under
"Confusion Matrix for Zero Shot" / "Confusion Matrix for Few Shot").

Predictions: `predictions/qwen_zeroshot_dev.csv`, `predictions/qwen_fewshot_dev.csv`
Scores: `results/qwen_zeroshot_scores/`, `results/qwen_fewshot_scores/`

---

## Summary: all systems

| System | Weighted F1 | SoftCons | HardCons |
|---|---|---|---|
| Trivial baseline | 0.1287 | 0.0 | 0.0 |
| DeBERTa-v3-base (failed) | 0.1287 | 0.0 | 0.0 |
| Qwen2.5-1.5B zero-shot | 0.2025 | 0.2130 | 0.0325 |
| Qwen2.5-1.5B few-shot | 0.4194 | 0.3357 | 0.2238 |
| DistilBERT (mean, 2 seeds) | 0.4349 | 0.3755 | 0.2708 |
