# Transformer-Enhanced LSTM for Darknet Traffic Classification

This is Phase 2 of a 3-phase Computer Networks project. Phase 1 built a CNN-LSTM
baseline. This phase builds an **improved model**, a Transformer-Enhanced LSTM,
on the CIC-Darknet2020 dataset. The model is an evolution of the Phase 1
baseline -- it keeps the CNN front-end and LSTM, then adds Transformer encoder
blocks (multi-head self-attention) before the classification head.

The model supports **three tasks**, selected with `--task`:

- **`binary`** -- Darknet vs Benign traffic, from the `Label` column. Phase 3
  will compare this against the Phase 1 baseline on the exact same held-out
  test rows.
- **`application`** -- 8-class application type (e.g. Browsing, P2P,
  Video-Streaming), from the `Label.1` column. This is a separate,
  additional task with no Phase 1 equivalent to compare against.
- **`fourclass`** -- a hierarchical extension of Task 1: trains one 4-class
  classifier on the same `Label` column (Tor, VPN, Non-Tor, NonVPN) and
  reports results at two levels, binary (grouped) and 4-class. See
  "Task 1 extension" below.

## Dataset

- Source: `dhoogla/cicdarknet2020` (Kaggle), same file used in Phase 1.
- Verified shape: **103,121 rows x 79 columns** (77 features + 2 label columns).
- No missing values, no infinite values, no duplicate rows in this build.
- Two label columns:
  - `Label` -- 4-class traffic type: `Non-Tor`, `NonVPN`, `VPN`, `Tor`.
    Note the exact spelling: `Non-Tor` is hyphenated, `NonVPN` is not.
    Used for the **binary** task.
  - `Label.1` -- 8 application types, with inconsistent casing in the raw
    data. Used for the **application** task, after normalization.
- Binary target (`--task binary`):
  - **Darknet (class 1)**: `Label` in `{Tor, VPN}` -- 18,101 rows (17.6%)
  - **Benign (class 0)**: `Label` in `{Non-Tor, NonVPN}` -- 85,020 rows (82.4%)
  - Because of this imbalance, **F1, recall, and the confusion matrix** are
    the headline metrics, not raw accuracy.
- Application target (`--task application`): `Label.1` has casing variants
  (e.g. `AUDIO-STREAMING`, `Video-streaming`, `File-transfer`) that collapse
  to 8 canonical classes (`config.APP_LABEL_CANON`, applied in
  `src/preprocessing.py`):

  | Class | Count |
  |---|---|
  | Browsing | 29,862 |
  | P2P | 23,404 |
  | Audio-Streaming | 11,328 |
  | File-Transfer | 10,647 |
  | Chat | 10,365 |
  | Video-Streaming | 9,012 |
  | Email | 5,442 |
  | VOIP | 3,061 |

  Imbalance ratio ~9.8x (Browsing vs VOIP) -- **macro-F1 and the per-class
  confusion matrix** are the headline metrics here, not overall accuracy.
- 15 constant (zero-variance) columns are dropped programmatically
  (`nunique() <= 1`), leaving ~62 usable features. They are not hardcoded;
  `src/preprocessing.py` detects and drops them at runtime and prints the list.
  This is identical for both tasks.
- The parquet file is committed to this repo (~12.8 MB) so that Colab can
  pull it via `git clone` -- no separate download step is required.

## Model: Transformer-Enhanced LSTM

Built with the Keras **Functional API** (required for the transformer
block's residual connections), via a single builder,
`build_transformer_lstm(n_features, n_classes, task)`, shared by both
tasks. Layer order:

1. `Conv1D(128, kernel=3, relu, same)` -> `MaxPool1D(2)` -> `Dropout(0.3)`
2. `Conv1D(64, kernel=3, relu, same)` -> `MaxPool1D(2)`
3. `LSTM(100, return_sequences=True)` (optionally `Bidirectional`, see
   below) -> `Dropout(0.3)`
4. **Transformer encoder block(s)** (multi-head self-attention +
   feed-forward, each with residual connection and LayerNorm), stacked
   `NUM_TRANSFORMER_BLOCKS` times (default 2) -- the enhancement over the
   Phase 1 baseline.
5. `GlobalAveragePooling1D()` to collapse the attended sequence to a vector
6. `Dense(64, relu)` -> `Dropout(0.3)` -> **task-specific output head**:
   - `binary`: `Dense(1, sigmoid)`, `binary_crossentropy` loss (or
     `BinaryFocalCrossentropy` if `LOSS_FN="focal"`).
   - `application`: `Dense(8, softmax)`, `sparse_categorical_crossentropy`
     loss (no one-hot encoding needed).

**What changed vs. the Phase 1 baseline:** the second sequence-modeling stage
is now Transformer encoder block(s) (multi-head self-attention) applied to the
LSTM's output sequence, followed by global average pooling, instead of a
second LSTM that compresses the sequence directly. This lets the model weigh
long-range relationships between flow features rather than relying only on
the LSTM's recurrent state.

As in the baseline: there is **no Embedding layer** (all inputs are
continuous tabular features), and the LSTM is a standard Keras `LSTM` layer.

**Optional architecture levers** (in `config.py`, modest defaults):
- `NUM_TRANSFORMER_BLOCKS` (default 2) -- stack more self-attention blocks.
- `USE_BILSTM` (default `False`) -- wrap the LSTM in `Bidirectional`
  (doubles its output width; the transformer block's residual/feed-forward
  dimensions adapt automatically since they're derived from the input
  shape).
- `LOSS_FN` (default `"bce"`) -- binary task only; `"focal"` switches to
  `BinaryFocalCrossentropy`, an alternative imbalance lever.
- `USE_CLASS_WEIGHT` (binary, default `False`) / `APP_USE_CLASS_WEIGHT`
  (application, default `True`) -- enable imbalance-aware training per task.

## Comparison integrity with Phase 1 (binary task only)

Everything except the model architecture is held identical to Phase 1, so
that Phase 3's comparison is apples-to-apples:

- Same dataset file and path, same binary label mapping.
- Same `preprocessing.py` logic: same constant-column drop, `StandardScaler`
  fit on the training split only, same reshape to `(n, n_features, 1)`.
- Same stratified split with `RANDOM_STATE=42`, `TEST_SIZE=0.20`,
  `VAL_SIZE=0.20` -- this means the held-out **test set is the exact same
  rows** as Phase 1.
- Same six metrics and `metrics.json` schema, so Phase 3 can load both
  phases' `metrics.json` files and tabulate them directly (the binary
  schema now nests metrics under `threshold_0_5` and `threshold_tuned`,
  see below).

### Binary fairness fix: matched training + threshold tuning

An earlier version of this model trained with `USE_CLASS_WEIGHT=True`,
which won on F1/AUC/recall but lost on precision/accuracy against the
Phase 1 baseline (trained without class weights) -- not an apples-to-apples
comparison, since the two models weren't trained the same way.

The **default is now matched to the baseline**: `USE_CLASS_WEIGHT=False`.
Instead of skewing the loss with class weights, the model's higher AUC is
cashed in via **validation-set threshold tuning** (`TUNE_THRESHOLD=True`):
`evaluate.py` sweeps thresholds on the validation set, picks the one that
maximizes F1, and applies it to the test set. Metrics are reported at
**both** the default 0.5 threshold and the tuned threshold, so neither
number is hidden -- expect the tuned threshold to trade some of the
lopsided recall for much better precision, landing on a more balanced
operating point.

To instead run the "best-result" class-weighted variant (a separate, valid
choice, not a substitute for threshold tuning), set `USE_CLASS_WEIGHT=True`
in `config.py` and describe the run accordingly in the report. Whichever
you choose, state it clearly; do not silently mix settings. See the
comment block at the top of `src/train.py` for the same note.

## Task 1 extension: hierarchical 4-class (`--task fourclass`)

Task 1 also has a hierarchical extension: `--task fourclass` trains one
4-class classifier on the same `Label` column, then `evaluate.py` reports
the result at **two levels**:

- **Level 1 (binary)** -- Darknet vs Benign, obtained by grouping the
  4-class prediction (Tor, VPN → Darknet; Non-Tor, NonVPN → Benign). This
  mirrors the base paper's own hierarchical labeling (Level 1 binary,
  Level 2 four-class).
- **Level 2 (4-class)** -- the full per-class result over Tor, VPN,
  Non-Tor, NonVPN.

The four classes nest exactly under the binary labels, so Level 1 is just
a grouping of Level 2. **Tor is rare (~1.1% of rows, ~55x imbalance vs
Non-Tor)**, so it is expected to be the hardest class -- class weights are
enabled by default for this task, and the headline Level-2 metrics are
**macro-F1 and the per-class confusion matrix**, not overall accuracy.
Level 1's numbers should land close to the dedicated `binary` task's
results (same coarse decision, routed through a 4-class model instead) --
the dedicated `binary` task remains the official binary result.

Run it the same way as the other tasks:
```bash
python src/train.py    --task fourclass
python src/evaluate.py --task fourclass
```
Outputs go to `results/fourclass/` and `figures/fourclass/`.

## How to run

### Local

This machine is not used for training. Only the source code lives here:
`config.py`, `src/preprocessing.py`, `src/model.py`, `src/train.py`,
`src/evaluate.py`. Nothing here has been fit or executed end-to-end.

### Colab

1. Push this repo to GitHub (see commands below).
2. Open `notebooks/colab_train.ipynb` in Google Colab.
3. Set `REPO_URL` in the second cell to this repo's URL.
4. Runtime -> Change runtime type -> GPU.
5. Runtime -> Run all. This clones the repo, installs light dependencies
   (pandas, scikit-learn, etc. -- TensorFlow is **not** reinstalled, to keep
   Colab's CUDA-matched build), then runs **all three tasks** end to end:
   ```bash
   python src/train.py    --task binary
   python src/evaluate.py --task binary
   python src/train.py    --task application
   python src/evaluate.py --task application
   python src/train.py    --task fourclass
   python src/evaluate.py --task fourclass
   ```
   and displays each task's figures and metrics inline. Outputs are kept
   separate per task under `results/binary/`, `results/application/`,
   `results/fourclass/`, `figures/binary/`, `figures/application/`,
   `figures/fourclass/` so the three runs never overwrite each other.

## Repository structure

```
transformer-enhanced-lstm-implementation/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── CIC-darknet2020-dataset/
│   └── cicdarknet2020.parquet
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── colab_train.ipynb
├── results/
│   ├── binary/
│   ├── application/
│   └── fourclass/
└── figures/
    ├── binary/
    ├── application/
    └── fourclass/
```

## Results

To be filled in after running `notebooks/colab_train.ipynb` on Colab.

**Binary (Darknet vs Benign)** -- report both rows; the tuned threshold is
the recommended operating point:

| Model                                 | Accuracy | F1  | Recall | Precision | AUC | Specificity |
|----------------------------------------|----------|-----|--------|-----------|-----|-------------|
| Transformer-Enhanced LSTM (thr=0.50)  | TBD      | TBD | TBD    | TBD       | TBD | TBD         |
| Transformer-Enhanced LSTM (thr=best)  | TBD      | TBD | TBD    | TBD       | TBD | TBD         |

With the matched setup (class weights off) plus tuned threshold, expect
the transformer to win or tie on **F1** at a balanced precision/recall
(not the lopsided high-recall/low-precision split seen under blind 0.5 +
class weighting) -- the better AUC is now spent at a better operating
point. Report F1 and AUC as the headline binary numbers.

**Application-type (8-class)** -- judge by macro-F1 and the confusion
matrix, not raw accuracy:

| Model                      | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Macro AUC |
|-----------------------------|----------|----------|-------------|------------------|--------------|-----------|
| Transformer-Enhanced LSTM  | TBD      | TBD      | TBD         | TBD              | TBD          | TBD       |

Overall accuracy will look lower than the binary task (8 classes, ~10x
imbalance) -- that's expected, not a regression. Expect small classes
(VOIP, Email) to be the hardest, and some confusion among the
streaming/browsing classes. A clear, honestly-reported per-class result
(see `results/application/classification_report.txt` and the confusion
matrix) is the deliverable, not a single inflated number.

**Fourclass (hierarchical Tor/VPN/Non-Tor/NonVPN)** -- report both levels:

| Level | Accuracy | F1 / Macro F1 | Recall / Macro Recall | Precision / Macro Precision | AUC / Macro AUC | Specificity |
|---|---|---|---|---|---|---|
| Level 1 (binary, grouped) | TBD | TBD | TBD | TBD | TBD | TBD |
| Level 2 (4-class)         | TBD | TBD | TBD | TBD | TBD | -- |

Level 1 should land close to the dedicated `binary` task's numbers -- it's
the same coarse decision, just routed through a 4-class model; small
differences are normal, and the dedicated `binary` task remains the
official binary result. Level 2's overall accuracy will read high because
Non-Tor dominates -- don't be fooled by that; judge it by **macro-F1** and
the confusion matrix (see `results/fourclass/classification_report.txt`).
Expect **Tor to be the weak class** (few samples) and some VPN/NonVPN or
Tor/Non-Tor confusion. A clean, honestly-reported per-class result, with
Tor visibly the hardest, is the deliverable.

## Team members

- TBD

## Academic integrity

All reported numbers come from the team's own Colab run on this exact
dataset version. Any numbers quoted from the reference paper are for
context only, not a substitute for this team's own results.
