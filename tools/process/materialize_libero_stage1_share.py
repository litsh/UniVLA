from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

from libero_process_balanced_mix import (
    build_manifest,
    choose_failure_counts,
    parse_dataset_spec,
    sample_indices,
)
from libero_process_stage1_pool import build_pool_episode_name


VIEW_DIRS = [
    "libero_all",
    "libero_all_codes_200",
    "libero_all_gripper_codes_200",
    "libero_all_birdview_codes_200",
    "libero_all_sideview_codes_200",
]
POOL_MANIFEST_FILES = [
    "libero_cot_manifest.json",
    "pool_manifest.json",
]


def make_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def materialize_subset(
    pool_root: Path,
    output_root: Path,
    keep_indices_by_alias: Dict[str, set[int]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for view_dir in VIEW_DIRS:
        src_view_root = pool_root / view_dir
        if not src_view_root.exists():
            continue
        dst_view_root = output_root / view_dir
        dst_view_root.mkdir(parents=True, exist_ok=True)

        for alias, selected_indices in keep_indices_by_alias.items():
            for episode_idx in sorted(selected_indices):
                scene_name = build_pool_episode_name(alias, episode_idx)
                src_scene = src_view_root / scene_name
                if not src_scene.exists():
                    raise FileNotFoundError(
                        f"Expected pool scene '{scene_name}' under {src_view_root}, but it does not exist."
                    )
                dst_scene = dst_view_root / scene_name
                make_symlink(src_scene, dst_scene)

    for filename in POOL_MANIFEST_FILES:
        src = pool_root / filename
        if src.exists():
            make_symlink(src, output_root / filename)


def main():
    parser = argparse.ArgumentParser(
        description="Materialize a new stage-1 failure-share dataset by symlinking from a full pool."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset spec in the format "
            "'<tfds_dir>|<suite>|<success_or_failure>|<original_or_occluded>|<alias>'."
        ),
    )
    parser.add_argument(
        "--pool_root",
        required=True,
        help="Root directory of the full processed+tokenized stage-1 pool.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Target dataset root to create, containing subset symlinks.",
    )
    parser.add_argument(
        "--target_failure_share",
        type=float,
        required=True,
        help="Target failure fraction in the materialized stage-1 dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--manifest_out",
        default=None,
        help="Optional JSON path to write the sampled subset manifest.",
    )
    args = parser.parse_args()

    specs = [parse_dataset_spec(raw_spec) for raw_spec in args.dataset]
    keep_counts = choose_failure_counts(specs, args.target_failure_share)
    keep_indices_by_alias = {
        spec.alias: sample_indices(spec.count, keep_counts.get(spec.alias, 0), args.seed, spec.alias)
        for spec in specs
    }
    manifest = build_manifest(
        specs=specs,
        keep_counts=keep_counts,
        keep_indices_by_alias=keep_indices_by_alias,
        target_failure_share=args.target_failure_share,
    )

    pool_root = Path(args.pool_root)
    output_root = Path(args.output_root)
    materialize_subset(pool_root=pool_root, output_root=output_root, keep_indices_by_alias=keep_indices_by_alias)

    manifest_path = Path(args.manifest_out) if args.manifest_out else output_root / "share_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
