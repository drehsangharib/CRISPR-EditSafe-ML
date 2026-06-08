# Real Public Dataset Workflow

This document describes how to move from the synthetic/demo data to a real public CRISPR off-target benchmark workflow.

## Recommended dataset sources

1. **CRISPRoffT**: comprehensive CRISPR/Cas off-target database with predicted and validated off-targets across many studies, technologies, cell contexts, and Cas/gRNA combinations.
2. **CRISPR-Net-compatible resources**: useful for sequence-pair off-target scoring with mismatches and indels.
3. **public_data_crisprCas9**: curated benchmark references and direct links to GUIDE-seq, CIRCLE-seq, SITE-seq, CHANGE-seq, and encoded benchmark files.

## Step 1: Put downloaded raw data in `data/raw/`

Example:

```powershell
mkdir data\raw
```

Manually download the public dataset file and place it in `data/raw/`. Avoid committing large raw datasets unless the license explicitly allows redistribution.

## Step 2: Audit the raw public table

```powershell
python scripts/audit_public_dataset.py --input data/raw/my_public_dataset.csv --output reports/my_public_dataset_audit.json --source-name my_public_dataset
```

The audit script prints:

- column names and dtypes
- first five records
- inferred guide/target/label/score columns
- cleaned sequence length distributions
- a suggested `prepare_public_dataset.py` command

## Step 3: Standardize to project schema

Use the command suggested by the audit script, for example:

```powershell
python scripts/prepare_public_dataset.py --input data/raw/my_public_dataset.csv --output data/processed/my_public_dataset_standardized.csv --source-name my_public_dataset --guide-col "on_seq" --target-col "off_seq" --score-col "CRISPR_Net_score" --score-threshold 0.1
```

## Step 4: Benchmark models

```powershell
python scripts/benchmark_models.py --input data/processed/my_public_dataset_standardized.csv --output-dir results/my_public_dataset_benchmark --models logistic_regression random_forest
```

## Step 5: Generate a Markdown benchmark report with figures

```powershell
python scripts/generate_benchmark_report.py --benchmark-dir results/my_public_dataset_benchmark --output reports/my_public_dataset_benchmark_report.md
```

## Step 6: Commit source files, not large generated files

Recommended files to commit:

- code under `crispr_editsafe/`
- scripts under `scripts/`
- documentation under `docs/`
- small example data under `data/examples/`

Recommended files not to commit:

- large raw datasets
- trained models
- generated benchmark outputs
- generated CSV/JSON result files
