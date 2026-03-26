"""Run pyInfinityFlow per donor to impute PE markers.

Usage:
    uv run python scripts/run_infinityflow.py [--donor D004] [--all]
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONORS = ["D004", "D005", "D006", "D007"]


def run_donor(donor: str):
    data_dir = os.path.join(ROOT, "data", "raw", donor)
    out_dir = os.path.join(ROOT, "data", "processed", f"infinityflow_{donor}")
    backbone_ann = os.path.join(ROOT, "data", "annotations", "backbone_annotation.csv")
    inf_ann = os.path.join(
        ROOT, "data", "annotations", f"infinity_marker_annotation_{donor}.csv"
    )

    for path in [data_dir, backbone_ann, inf_ann]:
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist. Run generate_annotations.py first.")
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyInfinityFlow.pyInfinityFlow",
        "--data_dir", data_dir,
        "--out_dir", out_dir,
        "--backbone_annotation", backbone_ann,
        "--infinity_marker_annotation", inf_ann,
        "--random_state", "42",
        "--n_events_combine", "500",
        "--use_logicle_scaling", "True",
        "--normalization_method", "zscore",
        "--save_h5ad", "True",
        "--verbosity", "1",
    ]

    print(f"\n{'='*60}")
    print(f"Running pyInfinityFlow for {donor}")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"ERROR: pyInfinityFlow failed for {donor} (exit code {result.returncode})")
        sys.exit(1)

    print(f"\nCompleted pyInfinityFlow for {donor}")


def main():
    parser = argparse.ArgumentParser(description="Run pyInfinityFlow per donor")
    parser.add_argument(
        "--donor", type=str, choices=DONORS, help="Run for a specific donor"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run for all donors sequentially"
    )
    args = parser.parse_args()

    if args.donor:
        run_donor(args.donor)
    elif args.all:
        for donor in DONORS:
            run_donor(donor)
    else:
        parser.print_help()
        print("\nSpecify --donor D004 or --all")


if __name__ == "__main__":
    main()
