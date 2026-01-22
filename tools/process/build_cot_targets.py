"""Utility to build chain-of-thought manifests for LIBERO-like episodes."""
import argparse
import json
import os
import sys
sys.path.append("/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA")
from tools.pickle_gen.pickle_generation_libero import build_reason_manifest


def main(dataset_root: str, suites, min_h: int, max_h: int, output: str) -> None:
    manifest = build_reason_manifest(dataset_root, suites, min_h, max_h)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved CoT manifest with {len(manifest)} episodes to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create CoT reasoning manifest for LIBERO episodes.")
    parser.add_argument("--dataset_root", default="data_storage/libero_all", help="Processed episode root.")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=[""],
        help="Subdirectories to include; empty string means dataset_root itself.",
    )
    parser.add_argument("--min_h", type=int, default=5, help="Lower bound of prediction horizon.")
    parser.add_argument("--max_h", type=int, default=10, help="Upper bound of prediction horizon.")
    parser.add_argument("--output", default="data_storage/libero_cot_manifest.json")
    args = parser.parse_args()

    suites = [suite for suite in args.suites if suite]
    if not suites:
        suites = [""]
    main(args.dataset_root.rstrip("/"), suites, args.min_h, args.max_h, args.output)
