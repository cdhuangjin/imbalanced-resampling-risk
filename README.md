# Imbalanced resampling risk

This repository contains source code, frozen analysis protocols, dataset manifests and hashes, run-level metrics, and figure source data for evaluating resampling strategies on public binary tabular benchmarks. It contains no manuscript files or exported publication figures.

## Environment and data

Use Python 3.12 or later and install the recorded dependencies:

```bash
python -m pip install -r requirements-reproduction.txt
python code/prepare_public_datasets.py
```

The preparation script fetches the public imbalanced-learn benchmark from Zenodo and verifies all 24 task shapes, class counts, and hashes. Set `SCIKIT_LEARN_DATA` if the default scikit-learn data home is unsuitable. The benchmark data themselves are not included here.

## Reproduce the main comparison

```bash
python code/run_main_full24.py --output code/outputs/recreated_main_full24
```

This runs the stratified 60/20/20 split, five seeds, training-fold-only resampling, and recorded learner settings. Full model runs can take substantial compute time.

## Archived metric records and figures

Archived `raw_results.csv` metric files are stored losslessly as `raw_results.csv.gz`. Restore them to their original names before running analyzers or regenerating figures from archived outputs:

```bash
python code/restore_compressed_metrics.py
```

The script does not overwrite existing uncompressed files. The figure-generation script and source-data CSVs are included; generated graphics are omitted and can be created locally with `python code/make_figures.py`.

Machine-specific absolute paths were removed from historical metadata. Dataset/cache locations are configured through `SCIKIT_LEARN_DATA` or the scikit-learn default data home.

## Checksums and licensing

`SHA256SUMS.txt` lists hashes for repository files other than the checksum list itself. The root MIT License applies to study-authored code in this repository; `official_code/` retains its separate MIT-0 license. Benchmark datasets and other third-party materials retain their upstream terms.
