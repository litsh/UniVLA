import os
import os.path as osp
import pickle, random
from glob import glob
from tqdm import tqdm
import numpy as np
import sys, json
import argparse
# Project-specific paths
PROJECT_ROOT = "/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli"
sys.path.append(f"{PROJECT_ROOT}/UniVLA")

# Import normalization utility
from train.dataset.normalize_pi0 import RunningStats, save

def sort_by_int(filename: str) -> int:
      stem = os.path.splitext(os.path.basename(filename))[0]
      try:
          return int(stem)
      except ValueError:
          # fall back to the raw stem so Python’s sort() still works
          return stem


def build_reason_manifest(language_dir: str, suites: list[str], min_h: int, max_h: int) -> dict[str, list[dict]]:
    """Build CoT annotations directly from processed LIBERO episodes."""
    manifest: Dict[str, List[Dict]] = {}
    suites = suites or [""]
    for suite in suites:
        episode_root = os.path.join(language_dir, suite) if suite else language_dir
        if not os.path.isdir(episode_root):
            continue
        for episode in tqdm(sorted(glob(os.path.join(episode_root, "*"))), desc=f"CoT {suite or 'libero'}"):
            frame_dir = os.path.join(episode, "images")
            frame_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")), key=sort_by_int)
            if len(frame_paths) < max_h:
                continue
            entries = []
            for idx in range(len(frame_paths) - min_h):
            # for idx in range(len(frame_paths) - 1):
                horizon = min(idx + random.randint(min_h, max_h), len(frame_paths) - 1)
                entries.append(
                    {
                        "obs_idx": idx,
                        "goal_idx": horizon,
                        "reasoning": f"Plan toward state {horizon - idx} steps ahead.",
                    }
                )
            if entries:
                manifest[os.path.basename(episode)] = entries
    return manifest


def main(
    dataset_path: str,
    output_path: str,
    normalizer_path: str,
    output_filename: str,
    manifest_path: str,
    min_h: int,
    max_h: int,
    suites: list[str],
) -> None:
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(normalizer_path, exist_ok=True)

    language_dir = osp.join(dataset_path, "libero_all")
    vq_dir = osp.join(dataset_path, "libero_all_codes_200")
    gripper_vq_dir = osp.join(dataset_path, "libero_all_gripper_codes_200")

    min_frames = 8
    result_file = []

    manifest = {}
    candidate = osp.join(dataset_path, manifest_path) if manifest_path else None
    if candidate and osp.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    # else:
    #     manifest = build_reason_manifest(language_dir, suites, min_h, max_h)

    print("Loading scenes from:", language_dir)
    for scene in tqdm(sorted(os.listdir(language_dir))):
        instr_file = osp.join(language_dir, scene, "instruction.txt")
        if not osp.exists(instr_file):
            continue
        with open(instr_file, "r") as f:
            text = f.read()

        action_folder = osp.join(language_dir, scene, "actions")
        if not osp.exists(action_folder):
            continue
        action_files = [osp.join(action_folder, file) for file in sorted(os.listdir(action_folder), key=sort_by_int)]
        if len(action_files) < min_frames:
            continue
        action = [np.load(a) for a in action_files]

        img_dir = osp.join(vq_dir, scene)
        if not osp.exists(img_dir):
            continue
        img_files = [osp.join(img_dir, file) for file in sorted(os.listdir(img_dir), key=sort_by_int)]

        gripper_img_dir = osp.join(gripper_vq_dir, scene)
        if not osp.exists(gripper_img_dir):
            continue
        gripper_img_files = [osp.join(gripper_img_dir, file) for file in sorted(os.listdir(gripper_img_dir), key=sort_by_int)]

        if len(img_files) < min_frames or len(gripper_img_files) < min_frames:
            continue

        cot_entries = []
        for entry in manifest.get(scene, []):
            goal_idx = entry.get("goal_idx", -1)
            if goal_idx < 0 or goal_idx >= len(img_files):
                continue
            cot_entries.append(
                {
                    "obs_idx": entry["obs_idx"],
                    "reasoning": entry.get("reasoning", ""),
                    "goal_tokens": [img_files[goal_idx]],
                }
            )
        if not cot_entries:
            continue

        result_file.append(
            {
                "text": text,
                "image": img_files,
                "action": action,
                "gripper_image": gripper_img_files,
                "reasoning": cot_entries,
            }
        )

    print(f"Total number of valid scenes: {len(result_file)}")
    if not result_file:
        raise ValueError("No valid scenes found. Check your dataset path.")

    normalizer = RunningStats()
    action_data = np.concatenate([scene["action"] for scene in result_file])
    normalizer.update(action_data)
    stats = normalizer.get_statistics()

    print("Mean:", stats.mean)
    print("Std:", stats.std)
    print("Q01:", stats.q01)
    print("Q99:", stats.q99)

    for scene in result_file:
        action = scene["action"]
        normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
        scene["action"] = np.clip(normalized, -1, 1)

    output_file = osp.join(output_path, output_filename)
    with open(output_file, "wb") as f:
        pickle.dump(result_file, f)
    print(f"Saved normalized data to {output_file}")

    save(normalizer_path, {"libero": stats})
    print(f"Saved normalizer statistics to {normalizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Libero dataset action values.")
    parser.add_argument("--dataset_path", type=str, default="data_storage/", help="Root path to dataset.")
    parser.add_argument("--output_path", type=str, default="data_storage/meta", help="Path to save normalized data.")
    parser.add_argument("--normalizer_path", type=str, default="configs/normalizer_libero", help="Path to save normalization stats.")
    parser.add_argument("--output_filename", type=str, default="libero_all_norm.pkl", help="Filename for normalized pickle output.")
    parser.add_argument("--cot_manifest", type=str, default="libero_cot_manifest.json", help="Manifest relative to dataset root.")
    parser.add_argument("--manifest_min_h", type=int, default=5)
    parser.add_argument("--manifest_max_h", type=int, default=10)
    parser.add_argument("--manifest_suites", nargs="+", default=[""])
    args = parser.parse_args()

    main(
        args.dataset_path,
        args.output_path,
        args.normalizer_path,
        args.output_filename,
        args.cot_manifest,
        args.manifest_min_h,
        args.manifest_max_h,
        args.manifest_suites,
    )