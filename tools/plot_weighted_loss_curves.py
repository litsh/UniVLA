import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


LOSS_KEYS = [
    "loss",
    "loss/groups_weighted_total",
    "loss/groups_unweighted_total",
    "loss/visual_content",
    "loss/visual_special",
    "loss/action_content",
    "loss/action_special",
    "learning_rate",
    "grad_norm",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot grouped loss curves from a Hugging Face trainer_state.json file."
    )
    parser.add_argument("trainer_state", help="Path to trainer_state.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <trainer_state_dir>/loss_curves.png",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Centered moving-average window for smoothed curves.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optionally downsample to at most this many points per curve. 0 disables downsampling.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    return parser.parse_args()


def load_history(path: str) -> List[Dict[str, float]]:
    with open(path) as f:
        state = json.load(f)
    history = state.get("log_history", [])
    rows = [row for row in history if "step" in row and any(key in row for key in LOSS_KEYS)]
    if not rows:
        raise ValueError(f"No step-based loss rows found in {path}")
    return rows


def parse_weights_from_path(path: str) -> Dict[str, float]:
    text = path
    patterns = {
        "loss/visual_content": r"VC=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "loss/visual_special": r"VS=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "loss/action_content": r"AC=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "loss/action_special": r"AS=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    }
    weights = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            weights[key] = float(match.group(1))
    return weights


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    radius = window // 2
    out = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        chunk = values[start:end]
        out.append(sum(chunk) / len(chunk))
    return out


def maybe_downsample(xs: List[float], ys: List[float], max_points: int) -> Tuple[List[float], List[float]]:
    if max_points <= 0 or len(xs) <= max_points:
        return xs, ys
    stride = int(math.ceil(len(xs) / max_points))
    return xs[::stride], ys[::stride]


def collect_series(rows: List[Dict[str, float]], key: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for row in rows:
        if key in row:
            xs.append(row["step"])
            ys.append(row[key])
    return xs, ys


def contribution_series(rows: List[Dict[str, float]], weights: Dict[str, float], key: str) -> Tuple[List[float], List[float]]:
    xs, ys = collect_series(rows, key)
    return xs, [weights.get(key, 1.0) * y for y in ys]


def final_value(rows: List[Dict[str, float]], key: str) -> Optional[float]:
    for row in reversed(rows):
        if key in row:
            return row[key]
    return None


def plot_series(ax, rows, key, label, color, smooth_window, max_points, linestyle="-", alpha=1.0):
    xs, ys = collect_series(rows, key)
    if not xs:
        return
    ys = moving_average(ys, smooth_window)
    xs, ys = maybe_downsample(xs, ys, max_points)
    ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, alpha=alpha)


def main():
    args = parse_args()
    rows = load_history(args.trainer_state)
    weights = parse_weights_from_path(args.trainer_state)

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.trainer_state), "loss_curves.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_total, ax_groups, ax_weighted, ax_opt = axes.flatten()

    plot_series(ax_total, rows, "loss", "trainer loss", "black", args.smooth_window, args.max_points)
    plot_series(
        ax_total,
        rows,
        "loss/groups_weighted_total",
        "group weighted total",
        "tab:red",
        args.smooth_window,
        args.max_points,
    )
    # plot_series(
    #     ax_total,
    #     rows,
    #     "loss/groups_unweighted_total",
    #     "group unweighted total",
    #     "tab:blue",
    #     args.smooth_window,
    #     args.max_points,
    # )
    ax_total.set_title("Aggregate Losses")
    ax_total.set_xlabel("step")
    ax_total.set_ylabel("loss")
    ax_total.grid(alpha=0.3)
    ax_total.legend()

    group_meta = [
        ("loss/visual_content", "visual content", "tab:green"),
        ("loss/visual_special", "visual special", "tab:olive"),
        ("loss/action_content", "action content", "tab:orange"),
        ("loss/action_special", "action special", "tab:purple"),
    ]
    for key, label, color in group_meta:
        plot_series(ax_groups, rows, key, label, color, args.smooth_window, args.max_points)
    ax_groups.set_title("Per-Group Losses")
    ax_groups.set_xlabel("step")
    ax_groups.set_ylabel("loss")
    ax_groups.set_yscale("log")
    ax_groups.grid(alpha=0.3)
    ax_groups.legend()

    for key, label, color in group_meta:
        xs, ys = contribution_series(rows, weights, key)
        if not xs:
            continue
        ys = moving_average(ys, args.smooth_window)
        xs, ys = maybe_downsample(xs, ys, args.max_points)
        weight = weights.get(key, 1.0)
        ax_weighted.plot(xs, ys, label=f"{label} x {weight:g}", color=color)
    ax_weighted.set_title("Weighted Group Contributions")
    ax_weighted.set_xlabel("step")
    ax_weighted.set_ylabel("weighted loss")
    ax_weighted.grid(alpha=0.3)
    ax_weighted.legend()

    plot_series(ax_opt, rows, "learning_rate", "learning rate", "tab:blue", args.smooth_window, args.max_points)
    ax_opt2 = ax_opt.twinx()
    xs, ys = collect_series(rows, "grad_norm")
    if xs:
        ys = moving_average(ys, args.smooth_window)
        xs, ys = maybe_downsample(xs, ys, args.max_points)
        ax_opt2.plot(xs, ys, label="grad norm", color="tab:red", alpha=0.7)
    ax_opt.set_title("Optimization Signals")
    ax_opt.set_xlabel("step")
    ax_opt.set_ylabel("learning rate", color="tab:blue")
    ax_opt2.set_ylabel("grad norm", color="tab:red")
    ax_opt.grid(alpha=0.3)

    title = args.title or os.path.basename(os.path.dirname(args.trainer_state))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")

    print(f"Saved plot to {output}")
    print("Final logged values:")
    for key in [
        "loss",
        "loss/groups_weighted_total",
        "loss/groups_unweighted_total",
        "loss/visual_content",
        "loss/visual_special",
        "loss/action_content",
        "loss/action_special",
    ]:
        value = final_value(rows, key)
        if value is not None:
            print(f"  {key}: {value:.6f}")
    if weights:
        print("Parsed weights:")
        for key, value in weights.items():
            print(f"  {key}: {value:g}")


if __name__ == "__main__":
    main()
