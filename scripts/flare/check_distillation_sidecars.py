#!/usr/bin/env python3
"""Validate FLARE distillation sidecars before launching GR00T training."""

from __future__ import annotations

import argparse
from pathlib import Path

from gr00t.data.dataset.flare_sidecars import resolve_flare_distillation_sidecars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distillation-dataset-dir",
        type=Path,
        default=Path("outputs/flare_distillation_filtered_direct"),
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=0,
        help="Optional exact record count; 0 disables the extra count assertion.",
    )
    parser.add_argument(
        "--expected-gt",
        type=int,
        default=-1,
        help="Optional exact GT count; a negative value disables this assertion.",
    )
    parser.add_argument(
        "--expected-teacher",
        type=int,
        default=-1,
        help="Optional exact teacher count; a negative value disables this assertion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecars = resolve_flare_distillation_sidecars(
        distillation_dataset_dir=args.distillation_dataset_dir,
        distillation_index_path=None,
        teacher_records_path=None,
    )
    if sidecars is None:
        raise RuntimeError("No FLARE sidecars were resolved")
    if args.expected_total > 0 and sidecars.total_records != args.expected_total:
        raise RuntimeError(
            f"FLARE record count is {sidecars.total_records}, expected {args.expected_total}"
        )
    gt_count = sidecars.source_counts.get("gt", 0)
    teacher_count = sidecars.source_counts.get("teacher", 0)
    if args.expected_gt >= 0 and gt_count != args.expected_gt:
        raise RuntimeError(f"FLARE GT count is {gt_count}, expected {args.expected_gt}")
    if args.expected_teacher >= 0 and teacher_count != args.expected_teacher:
        raise RuntimeError(
            f"FLARE teacher count is {teacher_count}, expected {args.expected_teacher}"
        )

    print("========== FLARE sidecar check ==========")
    print("index             :", sidecars.index_path)
    print("teacher records   :", sidecars.teacher_records_path)
    print("summary           :", sidecars.summary_path)
    print("total records     :", sidecars.total_records)
    print("source counts     :", sidecars.source_counts)
    print("status            : PASS")


if __name__ == "__main__":
    main()
