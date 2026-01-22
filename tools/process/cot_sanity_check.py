#!/usr/bin/env python
"""Sanity-check CoT-VLA training samples and evaluation inference.
Usage:
    python tools/process/cot_sanity_check.py sanity \
        --dataset /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/data_storage/meta/libero_all_norm.pkl \
        --emu_hub /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/logs/UNIVLA_LIBERO_CoTVLA_BS192_8k_gripper=False/checkpoint-8000 \
        --vq_hub /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-Stage1 \
        --vision_hub /inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/huggingface/Emu3-VisionTokenizer \
        --output cot_debug
"""

import argparse
import json
import os
import pickle, random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from glob import glob
import numpy as np
from PIL import Image
import torch

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "reference/RoboVLMs/eval/libero"))
from model_wrapper_emu import EmuVLAModel  # type: ignore

from transformers import AutoImageProcessor, AutoModel

def sort_by_int(filename: str) -> int:
    """Robust helper for sorting frame files even when given absolute paths."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(stem)
    except ValueError:
        # fall back to lexicographic ordering when the filename is not numeric
        return stem

def decode_goal(goal_paths, vision_hub, device, save_path):
    if not goal_paths:
        return None
    processor = AutoImageProcessor.from_pretrained(vision_hub, trust_remote_code=True)
    tokenizer = AutoModel.from_pretrained(vision_hub, trust_remote_code=True).to(device).eval()
    tensors = [torch.from_numpy(np.load(p)).unsqueeze(0).to(device) for p in goal_paths]
    codes = torch.cat(tensors, dim=1)
    with torch.no_grad():
        decoded = tokenizer.decode(codes)
    img = processor.postprocess(decoded)["pixel_values"][0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)
    return img


def make_observation(sample, obs_idx):
    token_path = Path(sample["image"][0])
    scene_name = token_path.parent.name
    token_root = token_path.parents[1]
    episode_dir = token_root.parent / "libero_all" / scene_name
    rgb_path = episode_dir / "images" / f"{obs_idx}.jpg"
    grip_path = episode_dir / "gripper_images" / f"{obs_idx}.jpg"
    observation = {
        "full_image": np.array(Image.open(rgb_path)),
        "state": np.zeros(7),
    }
    if grip_path.exists():
        observation["wrist_image"] = np.array(Image.open(grip_path))
    return observation, rgb_path, grip_path

def build_reason_manifest(language_dir: str, suites: List[str], min_h: int, max_h: int) -> Dict[str, List[Dict]]:
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
def run_sanity_check(args):
    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)
    sample = dataset[args.index]
    reasoning_list = sample.get("reasoning", [])
    if not reasoning_list:
        raise RuntimeError("Sample has no reasoning entries; dataset not CoT-prepared")
    entry = reasoning_list[0]
    obs_idx = entry.get("obs_idx", 0)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sample {args.index} instruction: {sample['text']}")
    print("Reasoning entry:", json.dumps(entry, indent=2))

    observation, rgb_path, grip_path = make_observation(sample, obs_idx)
    Image.open(rgb_path).save(out_dir / "observation.jpg")
    if grip_path.exists():
        Image.open(grip_path).save(out_dir / "gripper.jpg")

    goal_paths = entry.get("goal_tokens", [])
    if goal_paths:
        decode_goal(goal_paths, args.vision_hub, torch.device(args.device if torch.cuda.is_available() else "cpu"), out_dir / "dataset_goal.png")
        print("Saved dataset goal image -> dataset_goal.png")
    else:
        print("No goal token paths found in reasoning entry")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EmuVLAModel(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vision_hub,
        device=device,
        use_cot=True,
        cot_max_new_tokens=args.cot_max_new_tokens,
    )

    actions, reasoning = model.step(observation, sample["text"])
    np.save(out_dir / "pred_actions.npy", actions)
    print("Predicted actions saved to pred_actions.npy")
    if model.last_subgoal_image is not None:
        Image.fromarray(model.last_subgoal_image).save(out_dir / "pred_subgoal.png")
        print("Predicted subgoal saved to pred_subgoal.png")
    print("Model reasoning:", reasoning)


def main():
    parser = argparse.ArgumentParser(description="CoT manifest builder / sanity checker")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Create a CoT manifest")
    build.add_argument("--dataset_root", default="datasets/processed_data/libero_all")
    build.add_argument("--suites", nargs="+", default=[""])
    build.add_argument("--min_h", type=int, default=4)
    build.add_argument("--max_h", type=int, default=16)
    build.add_argument("--output", default="datasets/processed_data/libero_cot_manifest.json")

    sanity = subparsers.add_parser("sanity", help="Run sanity check on dataset + model")
    sanity.add_argument("--dataset", required=True)
    sanity.add_argument("--index", type=int, default=0)
    sanity.add_argument("--emu_hub", required=True)
    sanity.add_argument("--vq_hub", required=True)
    sanity.add_argument("--vision_hub", required=True)
    sanity.add_argument("--output", default="cot_debug")
    sanity.add_argument("--device", default="cuda")
    sanity.add_argument("--cot_max_new_tokens", type=int, default=256)

    args = parser.parse_args()
    if args.command == "build":
        suites = [suite for suite in args.suites if suite] or [""]
        manifest = build_reason_manifest(args.dataset_root.rstrip("/"), suites, args.min_h, args.max_h)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved CoT manifest with {len(manifest)} episodes to {args.output}")
    elif args.command == "sanity":
        run_sanity_check(args)
    else:
        parser.print_help()

    
if __name__ == "__main__":
    main()
