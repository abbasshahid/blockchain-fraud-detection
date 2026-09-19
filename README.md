# When Graph Learning Is Under-Supplied: Explainable Blockchain Fraud Detection with Evidence-Grounded LLMs

Reference implementation for the paper of the same name. The pipeline detects
illicit transactions in the Elliptic Bitcoin transaction graph under a
leakage-aware temporal protocol, and turns each prediction into an
investigation report whose every claim is checked against the evidence it was
given.

This is a **code-only** repository: it contains everything needed to *run* the
pipeline and nothing it produces. Tables, figures, metrics, trained models and
the manuscript are all regenerated locally by the commands below.

---

## What the paper finds

| Model             | Illicit F1                 | PR-AUC                     | ROC-AUC                    |
| ----------------- | -------------------------- | -------------------------- | -------------------------- |
| **XGBoost** | **0.6497 ± 0.0087** | **0.6630 ± 0.0020** | **0.8692 ± 0.0073** |
| GraphSAGE         | 0.2526 ± 0.0040           | 0.4289 ± 0.0405           | 0.8316 ± 0.0068           |
| GCN               | 0.2164 ± 0.0079           | 0.2090 ± 0.0155           | 0.7935 ± 0.0085           |
| GAT               | 0.1907 ± 0.0174           | 0.2203 ± 0.0512           | 0.7914 ± 0.0228           |

Five seeds (11, 22, 33, 44, 55), test period = time steps 40–49.

Three results worth highlighting:

- **The tabular baseline wins for a specific, measurable reason.** XGBoost keeps
  its advantage using only the 93 local features (F1 0.5184), which still beats
  every graph model given the full 165-dimensional vector. The graph models are
  limited by sparse labelled context: only 25.2% of test illicit transactions
  have an illicit neighbour at all.
- **The graph models' low precision is a threshold artefact.** A threshold
  chosen on the validation period alone lifts GraphSAGE from F1 0.2526 to 0.4136
  and cuts false positives from 2692 to 539.
- **Citation validity is not faithfulness.** A report can cite its evidence
  perfectly and still explain nothing, so the two are measured separately.
  Perturbing the detector shows roughly **61% of its decision comes from graph
  context** a feature-only model cannot see.

---

## Quick start

Requires Python 3.11+ and about 2 GB of free disk.

```bash
git clone https://github.com/abbasshahid/blockchain-fraud-detection.git
cd blockchain-fraud-detection
pip install -r requirements.lock.txt
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"
```

Download the three Elliptic CSV files and place them in
`elliptic_bitcoin_dataset/`:

```
elliptic_txs_features.csv
elliptic_txs_edgelist.csv
elliptic_txs_classes.csv
```

Then run the offline pipeline, which needs no API key:

```bash
make env             # record environment + pin requirements
make validate        # raw-data integrity checks
make preprocess      # leakage-aware temporal split
make train-all       # 4 models x 5 seeds       (~10 min, CPU)
make ablations       # local-only, alt splits, unknown-node  (~1 h, CPU)
make scalability     # cost and scaling curves
make evidence        # BTREP v2 evidence profiles
make artifacts       # all tables and figures
make check-figures   # fails if a panel is mis-sized or cropped
make test            # 16 tests
```

Or `make all` for the whole offline sequence.

Report generation with a language model is the only step that needs a key:

```bash
python -m blockchain_fraud.cli generate-reports --method grounded_llm      --llm-config configs/llm.yaml
python -m blockchain_fraud.cli generate-reports --method unconstrained_llm --llm-config configs/llm.yaml
```

`make reports-offline` produces deterministic template reports instead, with no
key at all. Keys are read locally and never written into any generated file.

| Provider      | Environment variable   | Key in`api_keys.json` |
| ------------- | ---------------------- | ----------------------- |
| Google Gemini | `GEMINI_API_KEY`     | `gemini_api_key`      |
| OpenRouter    | `OPENROUTER_API_KEY` | `openRouter_api_key`  |
| OpenAI        | `OPENAI_API_KEY`     | `openai_api_key`      |

Generation runs at temperature 0 and is resumable, so an interrupted run — an
exhausted free-tier quota, for instance — continues where it stopped.

---

## Layout

| Path                                 | Contents                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------ |
| `src/blockchain_fraud/data/`       | validation, temporal split, preprocessing                                            |
| `src/blockchain_fraud/models/`     | XGBoost baseline, GCN / GraphSAGE / GAT, graph utilities                             |
| `src/blockchain_fraud/training/`   | trainer and evaluator                                                                |
| `src/blockchain_fraud/blockchain/` | BTREP evidence profiles                                                              |
| `src/blockchain_fraud/explain/`    | prompts, LLM clients, schema, validators, faithfulness, evidence-value ablation      |
| `src/blockchain_fraud/analysis/`   | significance, thresholds, calibration, error and graph diagnostics, tables, figures  |
| `configs/`                         | one YAML per model, dataset variant and LLM provider —**the hyperparameters** |
| `scripts/`                         | environment report, ablations, scalability, artefact builds, figure checks           |

### What the pipeline creates locally

None of the following is tracked; each is produced by the commands above and
every writer creates its own directory, so a fresh clone needs no placeholders:

| Path                                                            | Created by                               | Size    |
| --------------------------------------------------------------- | ---------------------------------------- | ------- |
| `elliptic_bitcoin_dataset/`                                   | you, before anything else                | ~666 MB |
| `data/processed/`                                             | `make preprocess`                      | ~500 MB |
| `outputs/predictions/`, `outputs/checkpoints/`              | `make train-all`                       | ~850 MB |
| `outputs/tables/`, `outputs/metrics/`, `outputs/figures/` | `make artifacts`                       | ~4 MB   |
| `outputs/explanations/`                                       | `make evidence`, report generation     | ~1 MB   |
| `paper/`                                                      | the manuscript build                     | ~3 MB   |
| `docs/`                                                       | project notes written during development | <1 MB   |

Budget roughly 2 GB of free disk for a full run.

---

## Environment, seeds and hyperparameters

`requirements.lock.txt` pins the exact versions that produced the reported
numbers; `python scripts/00_environment_report.py` regenerates it and writes a
machine-readable record (package versions, CPU, thread count, dataset SHA-256
hashes, a snapshot of every config) to `outputs/metrics/environment.json`.

Reference environment: Python 3.13, PyTorch 2.10.0 (CPU build), XGBoost 3.1.3,
scikit-learn 1.7.2, SciPy 1.16.3, NumPy 2.3.5, pandas 2.3.3, matplotlib 3.10.6,
pydantic 2.12.4 — Windows 11, 12-thread Intel CPU, no GPU. Every model is
full-batch and CPU-only; no CUDA path is exercised.

**The hyperparameters are the files in `configs/`**, not a table in this README,
so they cannot drift from what the code runs:

| File                                                                | Role                                                              |
| ------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `configs/xgboost.yaml`                                            | 500 rounds, depth 6, lr 0.05, subsample 0.8, early stop 50        |
| `configs/gcn.yaml`, `configs/graphsage.yaml`                    | 32 hidden, dropout 0.35, lr 0.01, wd 5e-4, 60 epochs, patience 12 |
| `configs/gat.yaml`                                                | 24 hidden, dropout 0.35, lr 0.005, 50 epochs, patience 10         |
| `configs/data.yaml`                                               | 60/20/20 split over ordered time steps                            |
| `configs/data_local_only.yaml`                                    | local-only ablation: first 93 Elliptic columns                    |
| `configs/data_split_early.yaml`, `configs/data_split_late.yaml` | 50/20/30 and 70/15/15 boundaries                                  |
| `configs/llm.yaml`, `configs/llm_openrouter.yaml`               | provider, model, temperature, retry policy                        |

Model selection maximises validation PR-AUC in every case. Seeds 11, 22, 33, 44
and 55 are used everywhere; `blockchain_fraud.seed.set_seed` sets the Python,
NumPy and PyTorch generators, and XGBoost receives the same seed. Results are
reported as mean ± standard deviation over the five runs, and figures showing a
single run use seed 11.

Because all models share those five seeds, comparisons are paired: paired
*t*-tests with Holm correction within each metric, alongside Wilcoxon
signed-rank tests. With five pairs the two-sided Wilcoxon *p*-value cannot fall
below 0.0625, so it is reported as a sign-consistency check rather than a
powered test.

---

## Scientific guardrails

These are properties of the code, not conventions to remember:

- Unknown-label transactions stay in the graph as message-passing context but are
  excluded from the supervised loss, from threshold selection and from every
  reported metric.
- Feature scaling, class weights and motif thresholds are fitted on
  training-period nodes only.
- The decision threshold in the sensitivity analysis is selected on the
  validation period and applied **once** to the test period.
- BTREP evidence uses model probabilities and training-history label exposure
  only; it never reads hidden validation or test labels.
- The explanation layer cannot change a prediction — it consumes persisted
  outputs and nothing else.

Unit tests pin the temporal-split boundaries, their leakage properties, the
evidence-ID validator and the evidence-coverage definition.

---

## Intended use and limits

This pipeline estimates **transaction-level risk**; it does not establish that a
transaction is illicit, and it does not identify people or wallets. At the fixed
0.5 threshold the graph models raise thousands of alerts for a few hundred true
positives, so an alert is a request for human review and review capacity has to
be budgeted against the chosen operating point.

The evidence-ID validator guarantees that every claim in a generated report
points at supplied evidence. It does **not** guarantee that the transaction is
illicit, and a fluent report is persuasive out of proportion to its evidential
weight. Labels encode the priorities of one historical labelling process, so
error rates should be monitored per period and per transaction type rather than
only in aggregate. Results come from a single dataset covering one Bitcoin
period and do not transfer automatically to account-based or cross-chain
settings.

---

## Citation

```bibtex
@inproceedings{amjad2026explainable,
  title     = {When Graph Learning Is Under-Supplied: Explainable Blockchain Fraud Detection with Evidence-Grounded LLMs},
  author    = {Amjad, Sana and Abbas, Shahid and Shah, Syed Mohsin Ali and Taudes, Alfred},
  booktitle = {TODO: venue},
  year      = {2026}
}
```

The Elliptic dataset is credited to Weber et al., *Anti-Money Laundering in
Bitcoin: Experimenting with Graph Convolutional Networks for Financial
Forensics* (KDD Workshop on Anomaly Detection in Finance, 2019), and is subject
to its own terms; it is not redistributed here.

**MIT Licensed**
