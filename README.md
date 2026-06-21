# Transformer-Enhanced LSTM for Network Intrusion Detection

Phase 1 of this project built a CNN-LSTM baseline on CIC-Darknet2020. Phase 2
(this repo) built an **improved model**, a Transformer-Enhanced LSTM, on the
same dataset -- an evolution of the Phase 1 baseline that keeps the CNN
front-end and LSTM, then adds Transformer encoder blocks (multi-head
self-attention) before the classification head.

**Phase 3 re-targets the dataset-facing layer to CIC-IDS2017** -- a larger,
more standard network-intrusion-detection dataset -- without changing the
model architecture at all. Only `config.py`'s data-loading/label constants
and `src/preprocessing.py` changed; `src/model.py`'s `build_transformer_lstm`
is untouched. The CIC-Darknet2020 code paths (`--task binary/application/
fourclass`) are left in place for lineage but are now **inert locally** --
the original dataset parquet has been removed from this repo.

The model supports **six tasks** total, selected with `--task`:

**CIC-IDS2017 (Phase 3, current):**
- **`ids_binary`** -- Benign vs any attack, from `Label`. The one task that
  is genuinely class-comparable to the CIC-Darknet2020 `binary` result.
- **`ids_family`** -- coarse attack families (BENIGN + 7 buckets: DoS/DDoS,
  Brute-Force, Web-Attack, Botnet, PortScan, Infiltration, Other).
- **`ids_multi`** -- fine attack types (11 classes after merging the three
  Web Attack variants and dropping two ultra-rare classes).

**CIC-Darknet2020 (Phase 2, inert):**
- `binary`, `application`, `fourclass` -- code paths remain in `src/` for
  lineage but cannot run locally without the removed dataset file.

## Dataset (CIC-IDS2017)

- Source: `dhoogla/cicids2017` (Kaggle), "no-metadata" cleaned parquet --
  already stripped of Flow ID / source-destination IP / port / timestamp
  columns (verified below).
- Shipped as **8 day/attack-type files** under `CIC-IDS2017/`. All 8 are
  concatenated by `src/preprocessing.py` (`_load_ids2017`) -- using only
  some of them would silently drop whole classes (e.g. Heartbleed exists
  in exactly one file).

  | File | Rows |
  |---|---|
  | Benign-Monday-no-metadata.parquet | 458,831 |
  | Botnet-Friday-no-metadata.parquet | 176,038 |
  | Bruteforce-Tuesday-no-metadata.parquet | 389,714 |
  | DDoS-Friday-no-metadata.parquet | 221,264 |
  | DoS-Wednesday-no-metadata.parquet | 584,991 |
  | Infiltration-Thursday-no-metadata.parquet | 207,630 |
  | Portscan-Friday-no-metadata.parquet | 119,522 |
  | WebAttacks-Thursday-no-metadata.parquet | 155,820 |
  | **Combined** | **2,313,810** |

- **Combined shape: 2,313,810 rows x 78 columns** (77 numeric features + 1
  label column). All 8 files have identical columns -- no schema mismatch.
- Label column: **`Label`** (exact name, no surrounding whitespace), 15 raw
  classes:

  | Class | Count |
  |---|---|
  | Benign | 1,977,318 |
  | DoS Hulk | 172,846 |
  | DDoS | 128,014 |
  | DoS GoldenEye | 10,286 |
  | FTP-Patator | 5,931 |
  | DoS slowloris | 5,385 |
  | DoS Slowhttptest | 5,228 |
  | SSH-Patator | 3,219 |
  | PortScan | 1,956 |
  | Web Attack -- Brute Force | 1,470 |
  | Bot | 1,437 |
  | Web Attack -- XSS | 652 |
  | Infiltration | 36 |
  | Web Attack -- Sql Injection | 21 |
  | Heartbleed | 11 |

  **Two label-string gotchas, copy exactly, do not "fix":**
  - The benign string is **`"Benign"`**, not all-caps `"BENIGN"` -- the
    all-caps form is only used as the `ids_family` bucket name
    (`config.FAMILY_EXACT["Benign"] = "BENIGN"`).
  - The three `Web Attack` labels carry a **corrupted dash byte (`U+FFFD`)**
    between `"Web Attack"` and the variant name in the source parquet (a
    pre-existing artifact of the upstream dataset, not introduced here).
    All matching is done on the clean ASCII prefix `"Web Attack"`
    (`str.startswith`), which sidesteps the corrupted byte entirely.
- No missing values, no infinite values. **82,274 duplicate rows** are
  dropped during preprocessing (all `Benign`; the count is taken after the
  label column is dropped from the feature matrix, consistent with how the
  existing darknet pipeline checks duplicates -- a small number of rows
  share identical features but different labels, a known CICIDS2017 flow-
  export quirk).
- **No identifier/leakage columns** are present (no Flow ID, IPs, ports, or
  timestamp) -- already stripped by this "no-metadata" build. `config.
  IDS_DROP_COLS` is intentionally empty.
- **8 constant (zero-variance) columns** are dropped programmatically (same
  `nunique() <= 1` mechanism as the darknet pipeline, not hardcoded):
  `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`,
  `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`,
  `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`.
- **69 usable features** remain after dropping the label column and the 8
  constant columns.
- The 8 parquet files (~265 MB total) are committed to this repo so Colab
  can pull them via `git clone` -- no separate download step required.

## Tasks (CIC-IDS2017)

### `ids_binary` -- Benign vs Attack

`Label == "Benign"` &rarr; 0, anything else &rarr; 1.
- Benign: 1,895,314 rows (84.9%) after dedup
- Attack: 336,492 rows (15.1%) after dedup -- ~5.6x imbalance (milder than
  CIC-Darknet2020's Benign:Darknet ratio)

This is the **headline comparison** with the CIC-Darknet2020 `binary`
result -- same six metrics (accuracy, F1, recall, precision, AUC,
specificity), same threshold-tuning methodology (`config.TUNE_THRESHOLD`),
same split parameters. It's the one number that's genuinely apples-to-apples
across both datasets.

### `ids_family` -- coarse attack families

Each fine label is mapped to one of 8 buckets
(`config.FAMILY_EXACT` / `_map_family()` in `src/preprocessing.py`):

| Family | Count | Source labels |
|---|---|---|
| BENIGN | 1,977,318 | Benign |
| DoS/DDoS | 321,759 | DoS Hulk, DDoS, DoS GoldenEye, DoS slowloris, DoS Slowhttptest |
| Brute-Force | 9,150 | FTP-Patator, SSH-Patator |
| Web-Attack | 2,143 | Web Attack (Brute Force / XSS / Sql Injection) |
| PortScan | 1,956 | PortScan |
| Botnet | 1,437 | Bot |
| Infiltration | 36 | Infiltration |
| Other | 11 | Heartbleed (too rare -- 11 rows -- to stratify as its own family) |

### `ids_multi` -- fine attack types

The full fine-grained label set, with two cleanups before `LabelEncoder`:

1. The three `Web Attack ...` variants are merged into one clean `Web
   Attack` class (2,143 rows).
2. **Rare-class rule** (`config.MIN_CLASS_COUNT = 100`): a class with fewer
   rows is folded into its `ids_family` bucket if a same-family sibling
   survives at &ge; 100 rows in the fine label set (a real merge);
   otherwise its rows are dropped. In this dataset, **Infiltration (36
   rows)** and **Heartbleed (11 rows)** both have no surviving sibling at
   the fine level -- their families are singletons (`Infiltration` and
   `Other`/Heartbleed respectively have no other member) -- so folding
   would be a no-op and they are **dropped** instead (47 rows total).

Final 11 classes: `Benign`, `DoS Hulk`, `DDoS`, `DoS GoldenEye`,
`FTP-Patator`, `DoS slowloris`, `DoS Slowhttptest`, `SSH-Patator`,
`Web Attack`, `PortScan`, `Bot`.

### Honest comparison framing

Only **`ids_binary`** is class-comparable to a CIC-Darknet2020 result (the
dedicated `binary` task). **`ids_family`/`ids_multi`** demonstrate the
**same model architecture generalizing to a second dataset** on coarse/fine
multiclass intrusion detection -- their class-level numbers are **not**
comparable to the darknet `fourclass`/`application` classes, since the
problem domains are entirely different (encrypted-traffic-type
classification vs attack-type classification). The comparison story here is
"same model, second dataset, still works" -- not "same classes, different
dataset."

The dataset is highly imbalanced (Benign dominates `ids_family`/
`ids_multi`), and several rare attack classes were merged or dropped as
described above -- judge `ids_family`/`ids_multi` by **macro-F1** and the
confusion matrix, not raw accuracy. Expect the rarest surviving classes
(e.g. `Botnet`/`Bot`, `Infiltration`) to be the weak spots even after this
handling -- that's an honest, expected finding, not a bug.

If `config.SUBSAMPLE_N` is set for a given run, the reported numbers are on
a stratified majority-only-downsampled subsample (every minority/attack row
kept, only Benign downsampled) -- state this alongside any numbers quoted
from such a run.

## Model: Transformer-Enhanced LSTM

Built with the Keras **Functional API** (required for the transformer
block's residual connections), via a single builder,
`build_transformer_lstm(n_features, n_classes, task)`, shared by **all six
tasks** -- the architecture is completely dataset-agnostic; only
`n_features` (set by whichever preprocessing branch ran) and `n_classes`
(set by the task) vary. Layer order:

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
   - `binary` / `ids_binary`: `Dense(1, sigmoid)`, `binary_crossentropy`
     loss (or `BinaryFocalCrossentropy` if `LOSS_FN="focal"`).
   - `application` / `fourclass` / `ids_family` / `ids_multi`:
     `Dense(n_classes, softmax)`, `sparse_categorical_crossentropy` loss
     (no one-hot encoding needed).

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
- `LOSS_FN` (default `"bce"`) -- binary tasks only; `"focal"` switches to
  `BinaryFocalCrossentropy`, an alternative imbalance lever.
- `CLASS_WEIGHT_BY_TASK` -- per-task class-weight toggle: both binary tasks
  default to `False` (no class weight, threshold tuning instead);
  `application`/`fourclass`/`ids_family`/`ids_multi` default to `True`
  (`sklearn.compute_class_weight("balanced", ...)` in `train.py`).

## Comparison integrity with Phase 1 (CIC-Darknet2020 `binary` only)

Everything except the model architecture is held identical to Phase 1, so
that the darknet `binary` comparison is apples-to-apples:

- Same dataset file and path, same binary label mapping.
- Same `preprocessing.py` logic: same constant-column drop, `StandardScaler`
  fit on the training split only, same reshape to `(n, n_features, 1)`.
- Same stratified split with `RANDOM_STATE=42`, `TEST_SIZE=0.20`,
  `VAL_SIZE=0.20` -- this means the held-out **test set is the exact same
  rows** as Phase 1.
- Same six metrics and `metrics.json` schema, so Phase 3 can load both
  phases' `metrics.json` files and tabulate them directly.

### Binary fairness fix: matched training + threshold tuning

An earlier version of this model trained the darknet `binary` task with
`USE_CLASS_WEIGHT=True`, which won on F1/AUC/recall but lost on
precision/accuracy against the Phase 1 baseline (trained without class
weights) -- not an apples-to-apples comparison.

The **default is matched to the baseline**: `USE_CLASS_WEIGHT=False`.
Instead of skewing the loss with class weights, the model's higher AUC is
cashed in via **validation-set threshold tuning** (`TUNE_THRESHOLD=True`):
`evaluate.py` sweeps thresholds on the validation set, picks the one that
maximizes F1, and applies it to the test set. Metrics are reported at
**both** the default 0.5 threshold and the tuned threshold, so neither
number is hidden. The CIC-IDS2017 `ids_binary` task uses the **same
no-class-weight + tuned-threshold approach** by default
(`IDS_BINARY_USE_CLASS_WEIGHT=False`) -- both binary tasks are handled
identically.

## Task 1 extension: hierarchical 4-class (CIC-Darknet2020 `--task fourclass`)

The CIC-Darknet2020 `binary` task also has a hierarchical extension:
`--task fourclass` trains one 4-class classifier on the same `Label` column,
then `evaluate.py` reports the result at **two levels**:

- **Level 1 (binary)** -- Darknet vs Benign, obtained by grouping the
  4-class prediction (Tor, VPN → Darknet; Non-Tor, NonVPN → Benign).
- **Level 2 (4-class)** -- the full per-class result over Tor, VPN,
  Non-Tor, NonVPN.

This is inert locally now (dataset file removed), kept for lineage. Note
that `ids_family`/`ids_multi` do **not** get this two-level treatment --
they are reported as flat multiclass tasks via the shared
`evaluate_multiclass()`, since (unlike `fourclass` vs `binary`) they aren't
a finer-grained version of `ids_binary` sharing the same trained model.

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
   Colab's CUDA-matched build), then runs **all three CIC-IDS2017 tasks**
   end to end:
   ```bash
   python src/train.py    --task ids_binary
   python src/evaluate.py --task ids_binary
   python src/train.py    --task ids_family
   python src/evaluate.py --task ids_family
   python src/train.py    --task ids_multi
   python src/evaluate.py --task ids_multi
   ```
   and displays each task's figures and metrics inline. Outputs are kept
   separate per task under `results/ids_binary/`, `results/ids_family/`,
   `results/ids_multi/`, `figures/ids_binary/`, `figures/ids_family/`,
   `figures/ids_multi/` so the three runs never overwrite each other.

   **If training is slow** on this ~2.2M-row dataset, set `config.SUBSAMPLE_N`
   (e.g. `500_000`) before pushing -- it keeps every minority/attack row and
   only downsamples Benign, so rare classes aren't lost.

## Repository structure

```
transformer-enhanced-lstm-implementation/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── CIC-IDS2017/
│   ├── Benign-Monday-no-metadata.parquet
│   ├── Botnet-Friday-no-metadata.parquet
│   ├── Bruteforce-Tuesday-no-metadata.parquet
│   ├── DDoS-Friday-no-metadata.parquet
│   ├── DoS-Wednesday-no-metadata.parquet
│   ├── Infiltration-Thursday-no-metadata.parquet
│   ├── Portscan-Friday-no-metadata.parquet
│   └── WebAttacks-Thursday-no-metadata.parquet
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── colab_train.ipynb
├── results/
│   ├── ids_binary/
│   ├── ids_family/
│   └── ids_multi/
└── figures/
    ├── ids_binary/
    ├── ids_family/
    └── ids_multi/
```

## Results

To be filled in after running `notebooks/colab_train.ipynb` on Colab.

**`ids_binary` (Benign vs Attack)** -- the headline result, directly
comparable to CIC-Darknet2020's `binary` result; report both rows, the
tuned threshold is the recommended operating point:

| Model                                 | Accuracy | F1  | Recall | Precision | AUC | Specificity |
|----------------------------------------|----------|-----|--------|-----------|-----|-------------|
| Transformer-Enhanced LSTM (thr=0.50)  | TBD      | TBD | TBD    | TBD       | TBD | TBD         |
| Transformer-Enhanced LSTM (thr=best)  | TBD      | TBD | TBD    | TBD       | TBD | TBD         |

CIC-IDS2017 attacks tend to be fairly separable, so expect strong numbers
here -- this is the one result that's genuinely on the same page as the
CIC-Darknet2020 `binary` task. Report F1 and AUC as the headline numbers.

**`ids_family` (coarse attack families, 8 classes)** -- judge by macro-F1
and the confusion matrix, not raw accuracy:

| Model                      | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Macro AUC |
|-----------------------------|----------|----------|-------------|------------------|--------------|-----------|
| Transformer-Enhanced LSTM  | TBD      | TBD      | TBD         | TBD              | TBD          | TBD       |

Expect the rarest families (`Infiltration`, `Other`/Heartbleed) to be the
weak spots -- that's an honest, expected finding given 36 and 11 rows
respectively, not a regression.

**`ids_multi` (fine attack types, 11 classes)** -- judge by macro-F1 and the
confusion matrix, not raw accuracy:

| Model                      | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Macro AUC |
|-----------------------------|----------|----------|-------------|------------------|--------------|-----------|
| Transformer-Enhanced LSTM  | TBD      | TBD      | TBD         | TBD              | TBD          | TBD       |

`ids_family`/`ids_multi` are **not** comparable class-for-class to the
CIC-Darknet2020 `fourclass`/`application` results -- different problem
domain. The comparison story is "same model, second dataset, still works,"
not "same classes." If a run used `config.SUBSAMPLE_N`, say so here so the
numbers are interpreted correctly.

## Team members

- TBD

## Academic integrity

All reported numbers come from the team's own Colab run on this exact
dataset version. Any numbers quoted from the reference paper are for
context only, not a substitute for this team's own results.
