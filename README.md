# CRISPR-EditSafe-ML

**CRISPR-EditSafe-ML** is a GitHub-ready computational biology + machine learning project for CRISPR/Cas off-target risk prediction, guide safety ranking, reproducible model evaluation, explainability, and report generation.

> **Important:** This repository is for research, education, and portfolio demonstration only. It is **not** a clinical, diagnostic, therapeutic, or regulatory decision tool.

## Why this project exists

CRISPR therapeutics require careful assessment of potential off-target editing. This project demonstrates an end-to-end computational workflow that combines:

- CRISPR sequence feature engineering
- Off-target classification and ranking
- sgRNA-disjoint model validation
- Classical ML baselines
- Optional PyTorch deep learning model
- Explainability plots
- Automated testing
- Reproducible scripts and configs
- Scientific-style reporting

## Repository layout

```text
CRISPR-EditSafe-ML/
├── crispr_editsafe/          # Python package
├── configs/                  # YAML/JSON configs
├── data/examples/            # Tiny synthetic example dataset
├── notebooks/                # Placeholder notebook plan
├── reports/                  # Generated reports and figures
├── results/                  # Generated metrics and predictions
├── scripts/                  # Command-line scripts
├── tests/                    # Unit tests
├── workflows/                # Snakemake/Nextflow starter workflow
├── .github/workflows/        # GitHub Actions CI
├── Dockerfile
├── environment.yml
├── pyproject.toml
└── README.md
```

## Quick start

### Option 1: conda

```bash
conda env create -f environment.yml
conda activate crispr-editsafe-ml
pytest -q
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Run the complete example

The included example dataset is **synthetic toy data** for testing the pipeline mechanics.

```bash
python scripts/preprocess_data.py \
  --input data/examples/example_offtargets.csv \
  --output data/processed/features.csv

python scripts/train_model.py \
  --input data/processed/features.csv \
  --config configs/train_config.json \
  --model-out results/models/logreg_model.joblib \
  --metrics-out results/metrics/logreg_metrics.json \
  --predictions-out results/predictions/logreg_predictions.csv

python scripts/run_inference.py \
  --model results/models/logreg_model.joblib \
  --input data/examples/example_candidates.csv \
  --output results/predictions/example_candidate_predictions.csv

python scripts/generate_report.py \
  --metrics results/metrics/logreg_metrics.json \
  --predictions results/predictions/logreg_predictions.csv \
  --output reports/example_report.html
```

## Input format

Training input CSV must include:

```text
sgRNA,target,label
```

Optional columns:

```text
sample_id,source,pam,activity
```

Inference input CSV must include:

```text
sgRNA,target
```

## Features generated

The feature module generates interpretable CRISPR-pair features:

- mismatch count
- seed-region mismatch count
- mismatch counts by positional bins
- GC content of guide and target
- PAM proxy features
- per-position mismatch indicators
- one-hot encoded sgRNA and target bases

## Validation strategy

The default training script performs **sgRNA-disjoint splitting** when possible, meaning all rows for a guide are placed in either training or test, not both. This is more realistic than random row splitting because it tests whether the model can generalize to unseen guides.

## Example resume bullet

> Developed **CRISPR-EditSafe-ML**, a reproducible Python/scikit-learn pipeline for CRISPR off-target risk prediction using sequence feature engineering, sgRNA-disjoint validation, AUPRC/AUROC benchmarking, explainability plots, automated testing, and scientific-style HTML report generation.

## Limitations

- Included data are synthetic examples only.
- Public experimental datasets should be added before making scientific claims.
- This tool does not replace GUIDE-seq, CIRCLE-seq, CHANGE-seq, SITE-seq, targeted amplicon sequencing, or expert review.
- This tool is not validated for clinical decision-making.

## Next steps

Recommended extensions:

1. Add public GUIDE-seq/CIRCLE-seq benchmark datasets.
2. Add PyTorch CNN/Transformer model.
3. Add CRISPResso2-compatible output parser.
4. Add Snakemake or Nextflow production workflow.
5. Add model card and validation report.

## License

MIT License.


## Public dataset support

Version 2 adds a public-data adapter for CRISPR off-target benchmark tables. The adapter converts CSV/TSV/XLSX files from public resources into the internal schema:

```text
sample_id,source,sgRNA,target,label
```

Example:

```powershell
python scripts/prepare_public_dataset.py --input data/examples/public_schema_demo.csv --output data/processed/public_schema_demo_standardized.csv --source-name public_schema_demo --score-col CRISPR_Net_score --score-threshold 0.1
python scripts/benchmark_models.py --input data/processed/public_schema_demo_standardized.csv --output-dir results/public_benchmark --models logistic_regression random_forest
```

For real public datasets, place downloaded files under `data/raw/`, then use `scripts/prepare_public_dataset.py` with explicit `--guide-col`, `--target-col`, and either `--label-col` or `--score-col` if automatic column detection is not sufficient.


## Real public dataset workflow

Version 3 adds a real-dataset workflow layer:

```powershell
python scripts/audit_public_dataset.py --input data/raw/my_public_dataset.csv --output reports/my_public_dataset_audit.json --source-name my_public_dataset
python scripts/prepare_public_dataset.py --input data/raw/my_public_dataset.csv --output data/processed/my_public_dataset_standardized.csv --source-name my_public_dataset --guide-col "on_seq" --target-col "off_seq" --score-col "CRISPR_Net_score" --score-threshold 0.1
python scripts/benchmark_models.py --input data/processed/my_public_dataset_standardized.csv --output-dir results/my_public_dataset_benchmark --models logistic_regression random_forest
python scripts/generate_benchmark_report.py --benchmark-dir results/my_public_dataset_benchmark --output reports/my_public_dataset_benchmark_report.md
```

See `docs/REAL_DATA_WORKFLOW.md` for details.


## Public-data inference notes

The public-data adapter prioritizes biologically meaningful CRISPR sequence columns such as `guideSeq`, `otSeq`, `on_seq`, and `off_seq`, and activity columns such as `readFraction`. Numeric metadata fields such as mismatch counts, bulge counts, and guide specificity scores should not be inferred as sequence columns.
