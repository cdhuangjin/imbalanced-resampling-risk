from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from imblearn.datasets import fetch_datasets

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paperforge" / "supplementary" / "main_data_manifest_sha256.json"


def data_hash(x, y):
    digest = hashlib.sha256()
    digest.update(np.asarray(x, dtype=float).tobytes())
    digest.update(np.asarray(y).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Download the public imbalanced-learn benchmark data and verify the frozen task manifest."
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="use only an existing scikit-learn data cache")
    args = parser.parse_args()
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    loaded = fetch_datasets(
        filter_data=tuple(expected),
        download_if_missing=not args.verify_only,
        verbose=True,
    )
    failures = []
    records = {}
    for name, entry in expected.items():
        dataset = loaded[name]
        x = np.asarray(dataset["data"], dtype=float)
        y = np.asarray(dataset["target"])
        labels, counts = np.unique(y, return_counts=True)
        observed_counts = {str(label): int(count) for label, count in zip(labels, counts)}
        observed = {
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "class_counts": observed_counts,
            "data_hash": data_hash(x, y),
        }
        records[name] = observed
        if observed != entry:
            failures.append(name)
    result = {
        "verified": not failures,
        "dataset_count": len(records),
        "failed_tasks": failures,
        "data_home": "SCIKIT_LEARN_DATA or scikit-learn default",
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
