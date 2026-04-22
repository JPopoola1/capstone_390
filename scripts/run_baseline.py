from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import format_experiment_log, run_experiment, write_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Premier League xG travel baseline.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "raw" / "final_matches_distance.csv",
        help="Path to the prepared match-level CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory where metrics and runtime artifacts will be written.",
    )
    parser.add_argument(
        "--experiment-log-path",
        type=Path,
        default=ROOT / "logs" / "experiment_log.md",
        help="Markdown file that stores the latest experiment log entry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args.data_path, args.output_dir)
    write_results(results, args.output_dir)
    args.experiment_log_path.write_text(
        format_experiment_log(results),
        encoding="utf-8",
    )
    print(f"Wrote metrics to {args.output_dir / 'baseline_metrics.json'}")
    print(f"Wrote runtime budget to {args.output_dir / 'runtime_budget.json'}")
    print(f"Wrote experiment log to {args.experiment_log_path}")


if __name__ == "__main__":
    main()
