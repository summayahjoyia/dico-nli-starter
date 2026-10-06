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

### Part 1: Data exploration
All counts are from Track 1 (English). Code: `notebooks/dico_nli_work.ipynb`.

**Label distribution** (also saved in `results/part1/label_distribution.csv`)

| Label | Train | Train % | Dev | Dev % |
|---|---|---|---|---|
| FORWARD_ENTAILMENT | 881 | 29.0 | 190 | 28.8 |
| BACKWARD_ENTAILMENT | 881 | 29.0 | 190 | 28.8 |
| EQUIVALENCE | 802 | 26.4 | 174 | 26.4 |
| NEGATIVE_OTHER | 478 | 15.7 | 106 | 16.1 |
| Total | 3042 | 100 | 660 | 100 |

The labels are fairly balanced, and the train and dev proportions are almost the same. NEGATIVE_OTHER is the smallest class in both.

**Examples** (3 per label, with a reason for each): `results/part1/examples.csv`

**How reversed pairs are linked**
Each reversed pair is stored as two rows whose instance_id is share pair_id and end in __original or __flipped. In the dev reference file, the reverse_pair_id column holds the instance_id of the other row, so each twin points to the other. The twin has text1 and text2 swapped and its label reversed (FORWARD and BACKWARD are swapped but EQUIVALENCE stays the same). NEGATIVE_OTHER have no reverse(their reverse_pair_id is empty.)
