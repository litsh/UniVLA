#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_SUITES = ("goal", "object", "spatial")
SUMMARY_FILENAME = "summary.json"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)(?:_(.+))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate LIBERO offline perspective evaluation summaries across "
            "checkpoints into comparison tables."
        )
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("logs/libero/offline_perspective_metrics"),
        help="Root directory containing checkpoint subdirectories.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output file prefix. Defaults to <metrics-root>/checkpoint_comparison "
            "(without extension)."
        ),
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(DEFAULT_SUITES),
        help="Suites to aggregate, in output column order.",
    )
    parser.add_argument(
        "--sort-by",
        default="checkpoint_step",
        choices=(
            "checkpoint_step",
            "overall_weighted_token_accuracy",
            "overall_weighted_parse_success_rate",
            "overall_weighted_exact_match_rate",
            "overall_weighted_lpips_mean_valid_only",
            "overall_weighted_ssim_mean_valid_only",
        ),
        help="Column used to sort rows.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    return parser.parse_args()


def parse_checkpoint_name(name: str) -> Tuple[Optional[int], str]:
    match = CHECKPOINT_PATTERN.fullmatch(name)
    if not match:
        return None, ""
    step = int(match.group(1))
    variant = match.group(2) or ""
    return step, variant


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def find_checkpoint_dirs(metrics_root: Path, suites: Sequence[str]) -> List[Path]:
    checkpoint_dirs = []
    for child in sorted(metrics_root.iterdir()):
        if not child.is_dir():
            continue
        if parse_checkpoint_name(child.name)[0] is None:
            continue
        if any((child / suite / SUMMARY_FILENAME).exists() for suite in suites):
            checkpoint_dirs.append(child)
    return checkpoint_dirs


def load_checkpoint_metrics(checkpoint_dir: Path, suites: Sequence[str]) -> Dict:
    checkpoint_name = checkpoint_dir.name
    checkpoint_step, checkpoint_variant = parse_checkpoint_name(checkpoint_name)
    row = {
        "checkpoint": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "checkpoint_variant": checkpoint_variant,
        "checkpoint_path": str(checkpoint_dir.resolve()),
    }

    suite_summaries: Dict[str, Dict] = {}
    for suite in suites:
        summary_path = checkpoint_dir / suite / SUMMARY_FILENAME
        suite_prefix = f"{suite}_"
        row[f"{suite_prefix}summary_path"] = str(summary_path.resolve()) if summary_path.exists() else ""
        if summary_path.exists():
            summary = load_json(summary_path)
            suite_summaries[suite] = summary
            row[f"{suite_prefix}present"] = 1
            row[f"{suite_prefix}dataset_root"] = summary.get("dataset_root", "")
            row[f"{suite_prefix}num_samples"] = summary.get("num_samples")
            row[f"{suite_prefix}parse_success_rate"] = summary.get("parse_success_rate")
            row[f"{suite_prefix}token_accuracy"] = summary.get("token_accuracy")
            row[f"{suite_prefix}exact_match_rate"] = summary.get("exact_match_rate")
            row[f"{suite_prefix}lpips_mean_valid_only"] = summary.get("lpips_mean_valid_only")
            row[f"{suite_prefix}ssim_mean_valid_only"] = summary.get("ssim_mean_valid_only")
            row[f"{suite_prefix}num_lpips_valid"] = summary.get("num_lpips_valid")
            row[f"{suite_prefix}num_ssim_valid"] = summary.get("num_ssim_valid")
        else:
            row[f"{suite_prefix}present"] = 0
            row[f"{suite_prefix}dataset_root"] = ""
            row[f"{suite_prefix}num_samples"] = None
            row[f"{suite_prefix}parse_success_rate"] = None
            row[f"{suite_prefix}token_accuracy"] = None
            row[f"{suite_prefix}exact_match_rate"] = None
            row[f"{suite_prefix}lpips_mean_valid_only"] = None
            row[f"{suite_prefix}ssim_mean_valid_only"] = None
            row[f"{suite_prefix}num_lpips_valid"] = None
            row[f"{suite_prefix}num_ssim_valid"] = None

    add_overall_metrics(row, suite_summaries, suites)
    return row


def weighted_average(
    suite_summaries: Dict[str, Dict],
    suites: Sequence[str],
    metric_key: str,
    weight_key: str,
) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0
    for suite in suites:
        summary = suite_summaries.get(suite)
        if not summary:
            continue
        metric_value = summary.get(metric_key)
        weight_value = summary.get(weight_key)
        if metric_value is None or weight_value is None:
            continue
        weighted_sum += float(metric_value) * int(weight_value)
        total_weight += int(weight_value)
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def mean_average(
    suite_summaries: Dict[str, Dict],
    suites: Sequence[str],
    metric_key: str,
) -> Optional[float]:
    values = []
    for suite in suites:
        summary = suite_summaries.get(suite)
        if not summary:
            continue
        metric_value = summary.get(metric_key)
        if metric_value is not None:
            values.append(float(metric_value))
    if not values:
        return None
    return sum(values) / len(values)


def add_overall_metrics(row: Dict, suite_summaries: Dict[str, Dict], suites: Sequence[str]) -> None:
    total_num_samples = 0
    total_num_lpips_valid = 0
    total_num_ssim_valid = 0
    suites_present = 0
    for suite in suites:
        summary = suite_summaries.get(suite)
        if not summary:
            continue
        suites_present += 1
        total_num_samples += int(summary.get("num_samples", 0) or 0)
        total_num_lpips_valid += int(summary.get("num_lpips_valid", 0) or 0)
        total_num_ssim_valid += int(summary.get("num_ssim_valid", 0) or 0)

    row["num_suites_present"] = suites_present
    row["overall_total_num_samples"] = total_num_samples
    row["overall_total_num_lpips_valid"] = total_num_lpips_valid
    row["overall_total_num_ssim_valid"] = total_num_ssim_valid

    row["overall_weighted_parse_success_rate"] = weighted_average(
        suite_summaries, suites, "parse_success_rate", "num_samples"
    )
    row["overall_weighted_token_accuracy"] = weighted_average(
        suite_summaries, suites, "token_accuracy", "num_samples"
    )
    row["overall_weighted_exact_match_rate"] = weighted_average(
        suite_summaries, suites, "exact_match_rate", "num_samples"
    )
    row["overall_weighted_lpips_mean_valid_only"] = weighted_average(
        suite_summaries, suites, "lpips_mean_valid_only", "num_lpips_valid"
    )
    row["overall_weighted_ssim_mean_valid_only"] = weighted_average(
        suite_summaries, suites, "ssim_mean_valid_only", "num_ssim_valid"
    )

    row["overall_suite_mean_parse_success_rate"] = mean_average(
        suite_summaries, suites, "parse_success_rate"
    )
    row["overall_suite_mean_token_accuracy"] = mean_average(
        suite_summaries, suites, "token_accuracy"
    )
    row["overall_suite_mean_exact_match_rate"] = mean_average(
        suite_summaries, suites, "exact_match_rate"
    )
    row["overall_suite_mean_lpips_mean_valid_only"] = mean_average(
        suite_summaries, suites, "lpips_mean_valid_only"
    )
    row["overall_suite_mean_ssim_mean_valid_only"] = mean_average(
        suite_summaries, suites, "ssim_mean_valid_only"
    )


def build_columns(suites: Sequence[str]) -> List[str]:
    columns = [
        "checkpoint",
        "checkpoint_step",
        "checkpoint_variant",
        "num_suites_present",
        "overall_total_num_samples",
        "overall_total_num_lpips_valid",
        "overall_total_num_ssim_valid",
        "overall_weighted_parse_success_rate",
        "overall_weighted_token_accuracy",
        "overall_weighted_exact_match_rate",
        "overall_weighted_lpips_mean_valid_only",
        "overall_weighted_ssim_mean_valid_only",
        "overall_suite_mean_parse_success_rate",
        "overall_suite_mean_token_accuracy",
        "overall_suite_mean_exact_match_rate",
        "overall_suite_mean_lpips_mean_valid_only",
        "overall_suite_mean_ssim_mean_valid_only",
        "checkpoint_path",
    ]
    for suite in suites:
        prefix = f"{suite}_"
        columns.extend(
            [
                f"{prefix}present",
                f"{prefix}num_samples",
                f"{prefix}parse_success_rate",
                f"{prefix}token_accuracy",
                f"{prefix}exact_match_rate",
                f"{prefix}lpips_mean_valid_only",
                f"{prefix}ssim_mean_valid_only",
                f"{prefix}num_lpips_valid",
                f"{prefix}num_ssim_valid",
                f"{prefix}dataset_root",
                f"{prefix}summary_path",
            ]
        )
    return columns


def sort_rows(rows: List[Dict], sort_by: str, descending: bool) -> List[Dict]:
    def sort_key(row: Dict):
        value = row.get(sort_by)
        if sort_by == "checkpoint_step":
            checkpoint_step = row.get("checkpoint_step")
            checkpoint_variant = row.get("checkpoint_variant") or ""
            if checkpoint_step is None:
                return (float("inf"), checkpoint_variant, row["checkpoint"])
            return (int(checkpoint_step), checkpoint_variant, row["checkpoint"])
        if value is None:
            return (1, 0.0, row["checkpoint"])
        return (0, float(value), row["checkpoint"])

    return sorted(rows, key=sort_key, reverse=descending)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_value(row.get(column)) for column in columns})


def write_json(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2)


def build_markdown_table(rows: Sequence[Dict], suites: Sequence[str]) -> str:
    columns = [
        ("checkpoint", "checkpoint"),
        ("checkpoint_step", "step"),
        ("overall_weighted_token_accuracy", "overall_token"),
        ("overall_weighted_parse_success_rate", "overall_parse"),
        ("overall_weighted_lpips_mean_valid_only", "overall_lpips"),
        ("overall_weighted_ssim_mean_valid_only", "overall_ssim"),
    ]
    for suite in suites:
        columns.extend(
            [
                (f"{suite}_token_accuracy", f"{suite}_token"),
                (f"{suite}_parse_success_rate", f"{suite}_parse"),
                (f"{suite}_lpips_mean_valid_only", f"{suite}_lpips"),
                (f"{suite}_ssim_mean_valid_only", f"{suite}_ssim"),
            ]
        )

    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(format_value(row.get(column_name)) for column_name, _ in columns)
            + " |"
        )
    return "\n".join([header, separator] + body)


def write_markdown(path: Path, rows: Sequence[Dict], suites: Sequence[str], metrics_root: Path) -> None:
    lines = [
        "# LIBERO Offline Perspective Metrics",
        "",
        f"- Metrics root: `{metrics_root}`",
        f"- Checkpoints aggregated: `{len(rows)}`",
        f"- Suites: `{', '.join(suites)}`",
        "",
        "## Comparison Table",
        "",
        build_markdown_table(rows, suites),
        "",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    metrics_root = args.metrics_root.resolve()
    if not metrics_root.exists():
        raise FileNotFoundError(f"Metrics root does not exist: {metrics_root}")
    if not metrics_root.is_dir():
        raise NotADirectoryError(f"Metrics root is not a directory: {metrics_root}")

    output_prefix = args.output_prefix.resolve() if args.output_prefix else metrics_root / "checkpoint_comparison"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for checkpoint_dir in find_checkpoint_dirs(metrics_root, args.suites):
        rows.append(load_checkpoint_metrics(checkpoint_dir, args.suites))
    rows = sort_rows(rows, args.sort_by, args.descending)

    columns = build_columns(args.suites)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")

    write_csv(csv_path, columns, rows)
    write_json(json_path, rows)
    write_markdown(md_path, rows, args.suites, metrics_root)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
