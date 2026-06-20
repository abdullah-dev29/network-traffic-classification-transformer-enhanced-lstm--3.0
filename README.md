# Transformer-Enhanced LSTM for Darknet Traffic Classification

This is Phase 2 of a 3-phase Computer Networks project. Phase 1 built a CNN-LSTM
baseline. This phase builds an **improved model**, a Transformer-Enhanced LSTM,
for the same task: binary classification of network flows as **Darknet** or
**Benign** traffic on the CIC-Darknet2020 dataset. The model is an evolution of
the Phase 1 baseline -- it keeps the CNN front-end and LSTM, then adds a
Transformer encoder block (multi-head self-attention) before the classification
head. Phase 3 will compare this model against the Phase 1 baseline on the exact
same held-out test rows.

## Dataset

- Source: `dhoogla/cicdarknet2020` (Kaggle), same file used in Phase 1.
- Verified shape: **103,121 rows x 79 columns** (77 features + 2 label columns).
- No missing values, no infinite values, no duplicate rows in this build.
- Two label columns:
  - `Label` -- 4-class traffic type: `Non-Tor`, `NonVPN`, `VPN`, `Tor`.
    Note the exact spelling: `Non-Tor` is hyphenated, `NonVPN` is not.
  - `Label.1` -- 8 application types with inconsistent casing. **Not used** in
    this phase; it is dropped from the features. (If ever used, it needs
    casing normalization first.)
- Binary target used for training:
  - **Darknet (class 1)**: `Label` in `{Tor, VPN}` -- 18,101 rows (17.6%)
  - **Benign (class 0)**: `Label` in `{Non-Tor, NonVPN}` -- 85,020 rows (82.4%)
  - Because of this imbalance, **F1, recall, and the confusion matrix** are
    the headline metrics, not raw accuracy.
- 15 constant (zero-variance) columns are dropped programmatically
  (`nunique() <= 1`), leaving ~62 usable features. They are not hardcoded;
  `src/preprocessing.py` detects and drops them at runtime and prints the list.
- The parquet file is committed to this repo (~12.8 MB) so that Colab can
  pull it via `git clone` -- no separate download step is required.

## Model: Transformer-Enhanced LSTM

Built with the Keras **Functional API** (required for the transformer
block's residual connections). Layer order:

1. `Conv1D(128, kernel=3, relu, same)` -> `MaxPool1D(2)` -> `Dropout(0.3)`
2. `Conv1D(64, kernel=3, relu, same)` -> `MaxPool1D(2)`
3. `LSTM(100, return_sequences=True)` -> `Dropout(0.3)`
4. **Transformer encoder block** (multi-head self-attention + feed-forward,
   each with residual connection and LayerNorm) -- the enhancement over
   the Phase 1 baseline.
5. `GlobalAveragePooling1D()` to collapse the attended sequence to a vector
6. `Dense(64, relu)` -> `Dropout(0.3)` -> `Dense(1, sigmoid)`

**What changed vs. the Phase 1 baseline:** the second sequence-modeling stage
is now a Transformer encoder block (multi-head self-attention) applied to the
LSTM's output sequence, followed by global average pooling, instead of a
second LSTM that compresses the sequence directly. This lets the model weigh
long-range relationships between flow features rather than relying only on
the LSTM's recurrent state.

As in the baseline: there is **no Embedding layer** (all inputs are
continuous tabular features), and the LSTM is a standard Keras `LSTM` layer.

## Comparison integrity with Phase 1

Everything except the model architecture is held identical to Phase 1, so
that Phase 3's comparison is apples-to-apples:

- Same dataset file and path, same binary label mapping.
- Same `preprocessing.py` logic: same constant-column drop, `StandardScaler`
  fit on the training split only, same reshape to `(n, n_features, 1)`.
- Same stratified split with `RANDOM_STATE=42`, `TEST_SIZE=0.20`,
  `VAL_SIZE=0.20` -- this means the held-out **test set is the exact same
  rows** as Phase 1.
- Same `evaluate.py` metrics, 0.5 threshold, and `metrics.json` schema, so
  Phase 3 can load both phases' `metrics.json` files and tabulate them
  directly.

**Fair-Comparison Note:** this phase's training defaults
(`USE_CLASS_WEIGHT=True`, `EPOCHS=50` in `config.py`) are tuned for the
strongest result from the proposed model, which is one valid way to report
the comparison (describe it as "Transformer-Enhanced LSTM with
class-weighted training"). To isolate the contribution of the architecture
alone, set `USE_CLASS_WEIGHT=False` and `EPOCHS=30` in `config.py` to match
Phase 1's settings exactly -- then the only difference between the two
models is `model.py`. Whichever you choose, state it clearly in the report;
do not silently mix settings. See the comment block at the top of
`src/train.py` for the same note.

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
   Colab's CUDA-matched build), runs `src/train.py` then `src/evaluate.py`,
   and displays the resulting figures and metrics inline.

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
└── figures/
```

## Results

To be filled in after running `notebooks/colab_train.ipynb` on Colab.

| Model                      | Accuracy | F1  | Recall | Precision | AUC | Specificity |
|-----------------------------|----------|-----|--------|-----------|-----|-------------|
| Transformer-Enhanced LSTM  | TBD      | TBD | TBD    | TBD       | TBD | TBD         |

## Team members

- TBD

## Academic integrity

All reported numbers come from the team's own Colab run on this exact
dataset version. Any numbers quoted from the reference paper are for
context only, not a substitute for this team's own results.
