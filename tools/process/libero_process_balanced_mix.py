from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DEFAULT_VIEW_SPECS = [
    ("image", "images", "agentview_rgb"),
    ("gripper_image", "gripper_images", "eye_in_hand_rgb"),
    ("birdview_image", "birdview_images", "birdview_rgb"),
    ("sideview_image", "sideview_images", "sideview_rgb"),
]


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    suite: str
    kind: str
    domain: str
    alias: str
    count: int


def parse_view_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid view spec '{spec}'. Expected format: field_name:folder_name:observation_key"
        )
    return tuple(parts)


def parse_dataset_spec(raw_spec: str) -> DatasetSpec:
    parts = [part.strip() for part in raw_spec.split("|")]
    if len(parts) != 5:
        raise ValueError(
            "Each --dataset must use the format "
            "'<tfds_dir>|<suite>|<success_or_failure>|<original_or_occluded>|<alias>'."
        )

    path, suite, kind, domain, alias = parts
    if kind not in {"success", "failure"}:
        raise ValueError(f"Invalid kind '{kind}' in dataset spec '{raw_spec}'.")
    if domain not in {"original", "occluded"}:
        raise ValueError(f"Invalid domain '{domain}' in dataset spec '{raw_spec}'.")

    info_path = os.path.join(path, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Could not find dataset_info.json under {path}")

    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    split = next(s for s in data["splits"] if s["name"] == "train")
    count = sum(int(x) for x in split["shardLengths"])
    return DatasetSpec(path=path, suite=suite, kind=kind, domain=domain, alias=alias, count=count)


def build_episode_name(alias: str, file_path: str, global_index: int, path_depth: int) -> str:
    source = Path(file_path)
    parent_parts = list(source.parent.parts)
    tail = parent_parts[-path_depth:] if path_depth > 0 else []
    components = [alias]
    components.extend(part for part in tail if part)
    components.append(source.stem)
    components.append(str(global_index))
    return "__".join(components)


def compute_failure_budget(success_total: int, failure_share: float) -> int:
    if not (0.0 < failure_share < 1.0):
        raise ValueError("--target_failure_share must be between 0 and 1.")
    return round(success_total * failure_share / (1.0 - failure_share))


def allocate_suite_failure_quotas(
    success_by_suite: Dict[str, int],
    failure_capacity_by_suite: Dict[str, int],
    target_total_failures: int,
) -> Dict[str, int]:
    suite_quotas = {suite: 0 for suite in success_by_suite}
    remaining = set(success_by_suite)
    remaining_target = target_total_failures
    remaining_weight = sum(success_by_suite.values())

    while remaining and remaining_target > 0:
        changed = False
        for suite in list(remaining):
            if remaining_weight <= 0:
                break
            ideal = remaining_target * success_by_suite[suite] / remaining_weight
            add = min(int(round(ideal)), failure_capacity_by_suite[suite] - suite_quotas[suite])
            if add > 0:
                suite_quotas[suite] += add
                remaining_target -= add
                changed = True
            if suite_quotas[suite] >= failure_capacity_by_suite[suite]:
                remaining.remove(suite)
                remaining_weight -= success_by_suite[suite]

        if changed:
            continue

        candidates = []
        for suite in remaining:
            if suite_quotas[suite] < failure_capacity_by_suite[suite]:
                candidates.append(
                    (
                        suite_quotas[suite] / max(success_by_suite[suite], 1),
                        -failure_capacity_by_suite[suite],
                        suite,
                    )
                )
        if not candidates:
            break

        candidates.sort()
        _, _, suite = candidates[0]
        suite_quotas[suite] += 1
        remaining_target -= 1
        if suite_quotas[suite] >= failure_capacity_by_suite[suite]:
            remaining.remove(suite)
            remaining_weight -= success_by_suite[suite]

    return suite_quotas


def split_suite_quota_across_domains(
    dataset_specs: Iterable[DatasetSpec],
    suite_quota: int,
) -> Dict[str, int]:
    specs = sorted(dataset_specs, key=lambda spec: spec.domain)
    if len(specs) == 1:
        return {specs[0].alias: min(suite_quota, specs[0].count)}

    counts = {spec.alias: 0 for spec in specs}
    capacities = {spec.alias: spec.count for spec in specs}

    # Start with an even split to keep original and occluded balanced.
    for i in range(suite_quota):
        alias = specs[i % len(specs)].alias
        if counts[alias] < capacities[alias]:
            counts[alias] += 1

    assigned = sum(counts.values())
    remaining = suite_quota - assigned
    while remaining > 0:
        candidates = []
        for spec in specs:
            alias = spec.alias
            available = capacities[alias] - counts[alias]
            if available > 0:
                candidates.append((counts[alias], -available, alias))
        if not candidates:
            break
        candidates.sort()
        alias = candidates[0][2]
        counts[alias] += 1
        remaining -= 1

    return counts


def choose_failure_counts(
    dataset_specs: List[DatasetSpec],
    target_failure_share: float,
) -> Dict[str, int]:
    success_specs = [spec for spec in dataset_specs if spec.kind == "success"]
    failure_specs = [spec for spec in dataset_specs if spec.kind == "failure"]

    success_total = sum(spec.count for spec in success_specs)
    target_total_failures = compute_failure_budget(success_total, target_failure_share)

    success_by_suite = defaultdict(int)
    for spec in success_specs:
        success_by_suite[spec.suite] += spec.count

    failure_specs_by_suite = defaultdict(list)
    failure_capacity_by_suite = defaultdict(int)
    for spec in failure_specs:
        failure_specs_by_suite[spec.suite].append(spec)
        failure_capacity_by_suite[spec.suite] += spec.count

    suite_quotas = allocate_suite_failure_quotas(
        success_by_suite=success_by_suite,
        failure_capacity_by_suite=failure_capacity_by_suite,
        target_total_failures=target_total_failures,
    )

    keep_counts = {spec.alias: spec.count for spec in success_specs}
    for suite, specs in failure_specs_by_suite.items():
        split_counts = split_suite_quota_across_domains(specs, suite_quotas.get(suite, 0))
        keep_counts.update(split_counts)
    return keep_counts


def sample_indices(count: int, keep_count: int, seed: int, key: str) -> set[int]:
    if keep_count >= count:
        return set(range(count))
    rng = random.Random(f"{seed}:{key}")
    return set(rng.sample(range(count), keep_count))


def ensure_runtime_dependencies():
    import tensorflow as tf  # noqa: F401
    import tensorflow_datasets as tfds  # noqa: F401
    return tfds


def process_selected_episodes(
    specs: List[DatasetSpec],
    keep_indices_by_alias: Dict[str, set[int]],
    output_dir: str,
    extra_views: List[str],
    name_path_depth: int,
) -> None:
    tfds = ensure_runtime_dependencies()
    os.makedirs(output_dir, exist_ok=True)

    view_specs = list(DEFAULT_VIEW_SPECS)
    if extra_views:
        view_specs.extend(parse_view_spec(spec) for spec in extra_views)

    global_count = 0
    for spec in specs:
        selected_indices = keep_indices_by_alias.get(spec.alias, set())
        if not selected_indices:
            continue

        builder = tfds.builder_from_directory(spec.path)
        dataset = builder.as_dataset(split="train")

        for episode_idx, episode in enumerate(
            tqdm(dataset, desc=f"Processing {spec.alias}", total=spec.count, unit="episode")
        ):
            if episode_idx not in selected_indices:
                continue

            file_path = episode["episode_metadata"]["file_path"].numpy().decode()
            name = build_episode_name(spec.alias, file_path, global_count, name_path_depth)
            episode_dir = os.path.join(output_dir, name)
            os.makedirs(episode_dir, exist_ok=True)

            view_dirs = {}
            for _, folder_name, _ in view_specs:
                view_dir = os.path.join(episode_dir, folder_name)
                os.makedirs(view_dir, exist_ok=True)
                view_dirs[folder_name] = view_dir

            action_dir = os.path.join(episode_dir, "actions")
            os.makedirs(action_dir, exist_ok=True)

            view_images = {field_name: [] for field_name, _, _ in view_specs}
            languages = []
            actions = []

            for step in episode["steps"]:
                observation = step["observation"]
                action = step["action"]
                for field_name, _, observation_key in view_specs:
                    if observation_key not in observation:
                        raise KeyError(
                            f"Observation key '{observation_key}' not found for requested view '{field_name}'. "
                            f"Available keys: {list(observation.keys())}"
                        )
                    image = Image.fromarray(observation[observation_key].numpy())
                    view_images[field_name].append(image)

                languages.append(step["language_instruction"].numpy().decode())
                actions.append(action.numpy())

            for i in range(len(actions)):
                for field_name, folder_name, _ in view_specs:
                    view_images[field_name][i].save(os.path.join(view_dirs[folder_name], f"{i}.jpg"))
                np.save(os.path.join(action_dir, f"{i}.npy"), actions[i])
                if i == 0:
                    with open(os.path.join(episode_dir, "instruction.txt"), "w", encoding="utf-8") as f:
                        f.write(languages[i])

            global_count += 1


def build_manifest(
    specs: List[DatasetSpec],
    keep_counts: Dict[str, int],
    keep_indices_by_alias: Dict[str, set[int]],
    target_failure_share: float,
) -> dict:
    success_total = sum(spec.count for spec in specs if spec.kind == "success")
    kept_success_total = sum(
        keep_counts[spec.alias] for spec in specs if spec.kind == "success"
    )
    kept_failure_total = sum(
        keep_counts.get(spec.alias, 0) for spec in specs if spec.kind == "failure"
    )
    return {
        "target_failure_share": target_failure_share,
        "source_success_total": success_total,
        "kept_success_total": kept_success_total,
        "kept_failure_total": kept_failure_total,
        "actual_failure_share": kept_failure_total / max(kept_success_total + kept_failure_total, 1),
        "datasets": [
            {
                "alias": spec.alias,
                "path": spec.path,
                "suite": spec.suite,
                "kind": spec.kind,
                "domain": spec.domain,
                "available": spec.count,
                "kept": keep_counts.get(spec.alias, 0),
                "selected_indices": sorted(keep_indices_by_alias.get(spec.alias, set())),
            }
            for spec in specs
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Balanced LIBERO RLDS processor for stage-1 image training."
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
        "--output_dir",
        required=True,
        help="Directory where processed UniVLA episodes will be written.",
    )
    parser.add_argument(
        "--target_failure_share",
        type=float,
        default=0.15,
        help="Target fraction of failure episodes in the final processed dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic episode sampling.",
    )
    parser.add_argument(
        "--extra_view",
        action="append",
        default=[],
        help="Additional view spec in the format field_name:folder_name:observation_key.",
    )
    parser.add_argument(
        "--name_path_depth",
        type=int,
        default=2,
        help="How many trailing source path components to include in processed episode names.",
    )
    parser.add_argument(
        "--manifest_out",
        default=None,
        help="Optional path to save the selected-episode manifest JSON.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only compute and print the sampling plan without processing episodes.",
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

    print(json.dumps(manifest, indent=2))
    if args.manifest_out:
        os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
        with open(args.manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if args.dry_run:
        return

    process_selected_episodes(
        specs=specs,
        keep_indices_by_alias=keep_indices_by_alias,
        output_dir=args.output_dir,
        extra_views=args.extra_view,
        name_path_depth=args.name_path_depth,
    )


if __name__ == "__main__":
    main()
