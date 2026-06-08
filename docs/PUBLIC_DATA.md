# Public data upgrade

This project now includes a public-data adapter that converts downloaded CRISPR off-target tables into the internal training schema:

```text
sample_id,source,sgRNA,target,label
```

## Standardize a public dataset

Example with the included schema demo:

```powershell
python scripts/prepare_public_dataset.py --input data/examples/public_schema_demo.csv --output data/processed/public_schema_demo_standardized.csv --source-name public_schema_demo --score-col CRISPR_Net_score --score-threshold 0.1
```

If automatic column detection fails, specify columns explicitly:

```powershell
python scripts/prepare_public_dataset.py --input data/raw/my_public_dataset.csv --output data/processed/my_public_dataset_standardized.csv --source-name my_public_dataset --guide-col on_seq --target-col off_seq --score-col CRISPR_Net_score --score-threshold 0.1
```

## Benchmark models

```powershell
python scripts/benchmark_models.py --input data/processed/public_schema_demo_standardized.csv --output-dir results/public_benchmark --models logistic_regression random_forest
```

## Notes

- The adapter removes gap characters such as `_`, `-`, and `.`.
- The adapter converts `U` to `T`.
- The adapter keeps the first 20 nucleotides so the v1 feature encoder can be reused.
- If a table has a numeric activity/score column instead of binary labels, use `--score-col` and `--score-threshold`.
- Keep large downloaded files under `data/raw/` and avoid committing large datasets unless the license clearly permits redistribution.
