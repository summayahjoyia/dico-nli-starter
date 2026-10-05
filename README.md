# DiCo-NLI Starter Assignment (UR2PhD)

SemEval-2027 Task 2, Track 1 (English). All results are on the official dev split (660 items), scored with the official scorer. Train data is the official train split only. Dev data is never used for training.

## AI Assisstance
I used Claude to help me understand the pilot paper and to explain the starter kit code.

## Setup
- Platform: Google Colab, T4 GPU
- Task repo (data, scorer, starter kit): https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI
- Starter kit installed with `pip install -r requirements.txt` from `starter_kit/`

## How to reproduce

### Part 0: Setup and trivial baseline
1. Clone the task repo into `/content/task_repo`.
2. Predict the most frequent train label (FORWARD_ENTAILMENT, tied with BACKWARD_ENTAILMENT at 881) for every dev item and save as `instance_id,label`.
3. Score with:
   `python3 -m evaluation_functions --gold final_data/dev/dico_nli_dev_track1_reference.csv --predictions results/majority_dev.csv --output-dir results/majority_scores`

| System | Weighted F1 | SoftCons | HardCons |
|---|---|---|---|
| Always FORWARD_ENTAILMENT | 0.129 | 0.0 | 0.0 |

Files: `predictions/majority_dev.csv`, `results/majority_scores/`
