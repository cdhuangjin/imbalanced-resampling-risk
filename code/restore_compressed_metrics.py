#!/usr/bin/env python3
"""Restore compressed archived metric CSVs to analyzer-compatible filenames."""
from __future__ import annotations
import gzip
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    restored = skipped = 0
    for source in sorted(ROOT.rglob("raw_results.csv.gz")):
        target = source.with_suffix("")
        if target.exists():
            print(f"skip existing: {target.relative_to(ROOT)}")
            skipped += 1
            continue
        with gzip.open(source, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        print(f"restored: {target.relative_to(ROOT)}")
        restored += 1
    print(f"restored={restored} skipped={skipped}")
if __name__ == "__main__":
    main()
