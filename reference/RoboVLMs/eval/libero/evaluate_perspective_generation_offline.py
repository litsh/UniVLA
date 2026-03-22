"""
Offline evaluation for perspective-image generation on a fixed LIBERO rollout dataset.
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from model_wrapper_emu import EmuVLAModel
from libero_utils import save_rollout_gif


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline perspective-generation evaluation on a dumped LIBERO rollout dataset."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--emu_hub", type=str, required=True)
    parser.add_argument(
        "--vq_hub",
        type=str,
        default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-Base",
    )
    parser.add_argument(
        "--vision_hub",
        type=str,
        default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-VisionVQ",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_gripper", action="store_true")
    parser.add_argument(
        "--metric_image_size",
        type=int,
        default=200,
        help="Resize images to this square size before LPIPS/SSIM.",
    )
    parser.add_argument(
        "--qualitative_image_size",
        type=int,
        default=200,
        help="Display size for each panel in the qualitative GIFs.",
    )
    parser.add_argument("--qualitative_fps", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
    )
    return parser.parse_args()


def _to_uint8_image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _save_png(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(_to_uint8_image(image)).save(path)


def _resize_rgb(image, size):
    pil_image = Image.fromarray(_to_uint8_image(image))
    if pil_image.size != (size, size):
        pil_image = pil_image.resize((size, size), Image.BICUBIC)
    return np.asarray(pil_image)


def _image_to_tensor(image, size, device):
    image = _resize_rgb(image, size).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _build_gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_ssim(pred, target, window_size=11, sigma=1.5):
    if pred.shape != target.shape:
        raise ValueError(f"Expected matching shapes for SSIM, got {pred.shape} vs {target.shape}")

    channels = pred.shape[1]
    window = _build_gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return float(ssim_map.mean().item())


def _load_lpips_model(net_name, device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS evaluation requires the `lpips` package to be installed in the current environment."
        ) from exc

    metric = lpips.LPIPS(net=net_name).to(device)
    metric.eval()
    return metric


def _make_placeholder_image(size, text):
    image = Image.new("RGB", (size, size), color=(255, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), text, fill=(160, 0, 0))
    return np.asarray(image)


def _make_triptych_frame(input_agentview, gt_perspective, pred_perspective, caption, panel_size):
    images = [
        ("Input Agentview", _resize_rgb(input_agentview, panel_size)),
        ("GT Perspective", _resize_rgb(gt_perspective, panel_size)),
        ("Pred Perspective", _resize_rgb(pred_perspective, panel_size)),
    ]
    gutter = 12
    top_text_h = 26
    label_h = 22
    width = panel_size * 3 + gutter * 4
    height = top_text_h + panel_size + label_h + gutter
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 6), caption, fill=(0, 0, 0))

    y = top_text_h
    for idx, (label, image) in enumerate(images):
        x = gutter + idx * (panel_size + gutter)
        canvas.paste(Image.fromarray(image), (x, y))
        draw.text((x, y + panel_size + 4), label, fill=(0, 0, 0))
    return np.asarray(canvas)


def _relative_to(root, path):
    return os.path.relpath(path, root)


def load_dataset_records(dataset_root, max_samples=None):
    manifest_paths = sorted(Path(dataset_root).glob("samples_rank*.jsonl"))
    if not manifest_paths:
        raise FileNotFoundError(f"No dataset manifests found under {dataset_root}")

    records = []
    for manifest_path in manifest_paths:
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)

    records.sort(
        key=lambda item: (
            item["task_id"],
            item["episode_idx"],
            item["decision_step_idx"],
        )
    )
    if max_samples is not None:
        records = records[:max_samples]
    return records


def build_observation(sample, dataset_root):
    observation = {
        "full_image": _load_rgb(os.path.join(dataset_root, sample["input_agentview_path"])),
    }
    wrist_path = sample.get("input_wrist_path")
    if wrist_path:
        observation["wrist_image"] = _load_rgb(os.path.join(dataset_root, wrist_path))
    return observation


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)  # type: ignore

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_dataset_records(args.dataset_root, max_samples=args.max_samples)
    logger.info("Loaded %d offline evaluation samples from %s", len(records), args.dataset_root)

    model = EmuVLAModel(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vision_hub,
        device=device,
        use_perspective_gen=True,
        use_gripper=not args.no_gripper,
    )
    lpips_metric = _load_lpips_model(args.lpips_net, device)

    sample_metrics_path = os.path.join(args.output_dir, "per_sample_metrics.jsonl")
    sample_metrics_file = open(sample_metrics_path, "w")

    episode_frames = defaultdict(list)
    totals = {
        "num_samples": 0,
        "num_parse_success": 0,
        "token_accuracy_sum": 0.0,
        "exact_match_sum": 0.0,
        "lpips_sum": 0.0,
        "lpips_count": 0,
        "ssim_sum": 0.0,
        "ssim_count": 0,
    }

    with torch.no_grad():
        for sample in records:
            model.reset()
            observation = build_observation(sample, args.dataset_root)
            gt_perspective = _load_rgb(os.path.join(args.dataset_root, sample["gt_perspective_path"]))
            task_description = sample["task_description"]

            _ = model.step(observation, task_description)
            generation_info = model.get_last_generation_info()
            pred_token_grids = generation_info.get("perspective_token_grids", [])
            pred_images = generation_info.get("perspective_images", [])
            parse_success = bool(generation_info.get("parse_success", False)) and len(pred_token_grids) > 0

            gt_tokens = model.encode_visual_condition_image_tokens(gt_perspective)
            pred_tokens = pred_token_grids[0].cpu() if parse_success else None
            shape_match = pred_tokens is not None and tuple(pred_tokens.shape) == tuple(gt_tokens.shape)

            token_accuracy = 0.0
            exact_match = 0.0
            if shape_match:
                matches = pred_tokens.eq(gt_tokens)
                token_accuracy = float(matches.float().mean().item())
                exact_match = float(matches.all().item())

            lpips_value = None
            ssim_value = None
            if parse_success and pred_images:
                pred_image = pred_images[0]
                pred_metric_tensor = _image_to_tensor(pred_image, args.metric_image_size, device)
                gt_metric_tensor = _image_to_tensor(gt_perspective, args.metric_image_size, device)
                lpips_value = float(
                    lpips_metric(pred_metric_tensor * 2.0 - 1.0, gt_metric_tensor * 2.0 - 1.0).mean().item()
                )
                ssim_value = compute_ssim(pred_metric_tensor, gt_metric_tensor)
            else:
                pred_image = _make_placeholder_image(
                    args.qualitative_image_size, "parse failed"
                )

            pred_image_path = os.path.join(
                args.output_dir,
                "predicted_images",
                f"task{sample['task_id']:02d}",
                f"episode{sample['episode_idx'] + 1:03d}",
                f"step{sample['decision_step_idx']:04d}.png",
            )
            _save_png(pred_image, pred_image_path)

            caption = (
                f"task={sample['task_id']} ep={sample['episode_idx'] + 1} "
                f"step={sample['decision_step_idx']} "
                f"tok_acc={token_accuracy:.4f}"
            )
            if lpips_value is not None and ssim_value is not None:
                caption += f" lpips={lpips_value:.4f} ssim={ssim_value:.4f}"
            else:
                caption += " lpips=NA ssim=NA"

            qualitative_frame = _make_triptych_frame(
                observation["full_image"],
                gt_perspective,
                pred_image,
                caption=caption,
                panel_size=args.qualitative_image_size,
            )
            frame_path = os.path.join(
                args.output_dir,
                "qualitative_frames",
                f"task{sample['task_id']:02d}",
                f"episode{sample['episode_idx'] + 1:03d}",
                f"step{sample['decision_step_idx']:04d}.png",
            )
            _save_png(qualitative_frame, frame_path)

            episode_key = (sample["task_id"], sample["episode_idx"])
            episode_frames[episode_key].append((sample["decision_step_idx"], qualitative_frame))

            result = dict(sample)
            result.update(
                {
                    "parse_success": parse_success,
                    "num_predicted_frames": len(pred_token_grids),
                    "pred_token_shape": list(pred_tokens.shape) if pred_tokens is not None else None,
                    "gt_token_shape": list(gt_tokens.shape),
                    "shape_match": bool(shape_match),
                    "token_accuracy": token_accuracy,
                    "exact_match": exact_match,
                    "lpips": lpips_value,
                    "ssim": ssim_value,
                    "predicted_image_path": _relative_to(args.output_dir, pred_image_path),
                    "qualitative_frame_path": _relative_to(args.output_dir, frame_path),
                }
            )
            sample_metrics_file.write(json.dumps(result) + "\n")

            totals["num_samples"] += 1
            totals["num_parse_success"] += int(parse_success)
            totals["token_accuracy_sum"] += token_accuracy
            totals["exact_match_sum"] += exact_match
            if lpips_value is not None:
                totals["lpips_sum"] += lpips_value
                totals["lpips_count"] += 1
            if ssim_value is not None:
                totals["ssim_sum"] += ssim_value
                totals["ssim_count"] += 1

    sample_metrics_file.close()

    qualitative_gif_root = os.path.join(args.output_dir, "qualitative_gifs")
    os.makedirs(qualitative_gif_root, exist_ok=True)
    for (task_id, episode_idx), frames in episode_frames.items():
        frames.sort(key=lambda item: item[0])
        gif_frames = [frame for _, frame in frames]
        gif_path = os.path.join(
            qualitative_gif_root,
            f"task{task_id:02d}",
            f"episode{episode_idx + 1:03d}.gif",
        )
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        save_rollout_gif(gif_frames, gif_path, fps=args.qualitative_fps)

    summary = {
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "num_samples": totals["num_samples"],
        "parse_success_rate": (
            totals["num_parse_success"] / totals["num_samples"] if totals["num_samples"] > 0 else 0.0
        ),
        "token_accuracy": (
            totals["token_accuracy_sum"] / totals["num_samples"] if totals["num_samples"] > 0 else 0.0
        ),
        "exact_match_rate": (
            totals["exact_match_sum"] / totals["num_samples"] if totals["num_samples"] > 0 else 0.0
        ),
        "lpips_mean_valid_only": (
            totals["lpips_sum"] / totals["lpips_count"] if totals["lpips_count"] > 0 else None
        ),
        "ssim_mean_valid_only": (
            totals["ssim_sum"] / totals["ssim_count"] if totals["ssim_count"] > 0 else None
        ),
        "num_lpips_valid": totals["lpips_count"],
        "num_ssim_valid": totals["ssim_count"],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
