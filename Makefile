.PHONY: env validate preprocess train-one train-all ablations scalability evidence reports-offline artifacts check-figures numbers test all

PY := python
SEEDS := 11,22,33,44,55
EVIDENCE := outputs/explanations/elliptic_temporal_v1_graphsage_all_original_features_seed11_btrep_v2.json

env:
	$(PY) scripts/00_environment_report.py

validate:
	$(PY) -m blockchain_fraud.cli validate-data --raw-dir elliptic_bitcoin_dataset

preprocess:
	$(PY) -m blockchain_fraud.cli preprocess --config configs/data.yaml

train-one:
	$(PY) -m blockchain_fraud.cli train --model xgboost --seed 11 --config configs/xgboost.yaml

train-all:
	$(PY) -m blockchain_fraud.cli train-all --models xgboost,gcn,graphsage,gat --seeds $(SEEDS)

ablations:
	$(PY) scripts/10_run_ablations.py --family all --seeds $(SEEDS)

scalability:
	$(PY) scripts/11_scalability.py

evidence:
	$(PY) -m blockchain_fraud.cli generate-evidence --model graphsage --seed 11 --limit 100 --evidence-version v2

# Deterministic reports; needs no API key.
reports-offline:
	$(PY) -m blockchain_fraud.cli generate-reports --evidence-path $(EVIDENCE) --method template

artifacts:
	$(PY) scripts/12_build_revision_artifacts.py

# Fails if the paper panels differ in size or crop their content.
check-figures:
	$(PY) scripts/15_check_figures.py

numbers:
	$(PY) scripts/13_paper_numbers.py

test:
	pytest -q

# Full offline reproduction, excluding the API-dependent report generation.
all: env validate preprocess train-all ablations scalability evidence reports-offline artifacts check-figures numbers
