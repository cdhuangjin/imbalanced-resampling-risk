"""Analyze the confidence-guarded branch with the frozen adaptive Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_adaptive import analyze_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze_gate(args.input, args.output)
    protocol_path = args.input / "confidence_guarded_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["selection_rule"] = protocol["name"]
    payload["confidence_guarded_protocol"] = str(protocol_path.resolve())
    (args.output / "gate_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
