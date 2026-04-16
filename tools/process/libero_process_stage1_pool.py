from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

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


def ensure_runtime_dependencies():
    import tensorflow_datasets as tfds  # noqa: F401
    return tfds


def parse_view_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid view spec '{spec}'. Expected format: field_name:folder_name:observation_key"
        )
    return tuple(parts)


def parse_dataset_spec(raw_spec: str) -> tuple[str, str]:
    parts = [part.strip() for part in raw_spec.split("|")]
    if len(parts) != 5:
        raise ValueError(
            "Each --dataset must use the format "
            "'<tfds_dir>|<suite>|<success_or_failure>|<original_or_occluded>|<alias>'."
        )
    path, _, _, _, alias = parts
    return path, alias


def build_pool_episode_name(alias: str, episode_idx: int) -> str:
    return f"{alias}__episode_{episode_idx:06d}"


def process_dataset(
    dataset_dir: str,
    alias: str,
    output_dir: str,
    extra_views: List[str],
) -> int:
    tfds = ensure_runtime_dependencies()
    view_specs = list(DEFAULT_VIEW_SPECS)
    if extra_views:
        view_specs.extend(parse_view_spec(spec) for spec in extra_views)

    builder = tfds.builder_from_directory(dataset_dir)
    dataset = builder.as_dataset(split="train")

    count = 0
    for episode_idx, episode in enumerate(
        tqdm(dataset, desc=f"Processing {alias}", unit="episode")
    ):
        name = build_pool_episode_name(alias, episode_idx)
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

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Process all stage-1 RLDS sources into a reusable stable-named pool."
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
        help="Output directory for processed episodes, usually <pool_root>/libero_all.",
    )
    parser.add_argument(
        "--extra_view",
        action="append",
        default=[],
        help="Additional view spec in the format field_name:folder_name:observation_key.",
    )
    parser.add_argument(
        "--manifest_out",
        default=None,
        help="Optional JSON path to save pool metadata.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    manifest = {"datasets": []}
    for raw_spec in args.dataset:
        dataset_dir, alias = parse_dataset_spec(raw_spec)
        count = process_dataset(
            dataset_dir=dataset_dir,
            alias=alias,
            output_dir=args.output_dir,
            extra_views=args.extra_view,
        )
        manifest["datasets"].append(
            {
                "alias": alias,
                "dataset_dir": dataset_dir,
                "count": count,
            }
        )

    if args.manifest_out:
        os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
        with open(args.manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
