import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DEFAULT_VIEW_SPECS = [
    ("image", "images", "agentview_rgb"),
    ("gripper_image", "gripper_images", "eye_in_hand_rgb"),
    ("birdview_image", "birdview_images", "birdview_rgb"),
    ("sideview_image", "sideview_images", "sideview_rgb")
]


def parse_view_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid view spec '{spec}'. Expected format: field_name:folder_name:observation_key"
        )
    return tuple(parts)


def main(dataset_dirs: str, base_output_dir: str, extra_views: list[str]) -> None:
    builder = tfds.builder_from_directory(dataset_dirs)
    ds_all_dict = builder.as_dataset(split="train")

    os.makedirs(base_output_dir, exist_ok=True)

    view_specs = list(DEFAULT_VIEW_SPECS)
    if extra_views:
        view_specs.extend(parse_view_spec(spec) for spec in extra_views)

    count = 0
    for episode in tqdm(ds_all_dict, desc="Processing episodes", unit="episode"):
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        name = file_path.split("/")[-2] + "__" + file_path.split("/")[-1].split(".")[0] + "__" + str(count)

        episode_dir = os.path.join(base_output_dir, name)
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

        for i, step in tqdm(
            enumerate(episode["steps"]),
            desc=f"Processing episode {name}",
            total=len(episode["steps"]),
            unit="step",
        ):
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

            language = step["language_instruction"].numpy().decode()
            languages.append(language)
            actions.append(action)

        for i in range(len(actions)):
            for field_name, folder_name, _ in view_specs:
                view_images[field_name][i].save(os.path.join(view_dirs[folder_name], f"{i}.jpg"))
            np.save(os.path.join(action_dir, f"{i}.npy"), actions[i].numpy())
            if i == 0:
                with open(os.path.join(episode_dir, "instruction.txt"), "w") as f:
                    f.write(languages[i])

        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LIBERO episodes into image/action folders.")
    parser.add_argument(
        "--dataset_dirs",
        default="/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/openvla-oft/data_storage/original_libero_multiview/libero_object/1.0.0",
        help="TFDS directory for the LIBERO dataset split.",
    )
    parser.add_argument(
        "--output_dir",
        default="/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/data_storage/libero_all",
        help="Directory to save processed LIBERO episodes.",
    )
    parser.add_argument(
        "--extra_view",
        action="append",
        default=[],
        help="Additional view spec in the format field_name:folder_name:observation_key. "
             "Example: birdview_image:birdview_images:birdview_rgb",
    )
    args = parser.parse_args()

    main(args.dataset_dirs, args.output_dir, args.extra_view)
