from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_experiment import run_slice

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "abalone", "abalone_19", "arrhythmia", "car_eval_34", "car_eval_4",
    "coil_2000", "ecoli", "letter_img", "libras_move", "mammography", "oil",
    "optical_digits", "ozone_level", "pen_digits", "satimage", "scene",
    "sick_euthyroid", "solar_flare_m0", "spectrometer", "thyroid_sick",
    "us_crime", "wine_quality", "yeast_me2", "yeast_ml8",
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the frozen 24-task DT/CatBoost main comparison."
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "code" / "outputs" / "recreated_main_full24",
    )
    args = parser.parse_args()
    status = run_slice(
        args.output,
        datasets=DATASETS,
        learners=("dt", "cat"),
        methods=("none", "random", "smote"),
        seeds=(0, 1, 2, 3, 4),
    )
    print(json.dumps(status, indent=2))
    return 0 if status.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
