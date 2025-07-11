"""Main launcher script for PBT experiments."""

import argparse
import importlib
import sys
import os

from pbt_baseline.launcher.run_processes import add_os_parallelism_args, run
from pbt_baseline.launcher.run_slurm import add_slurm_args, run_slurm


def launcher_argparser(args) -> argparse.ArgumentParser:
    """Create argument parser for launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./train_dir", type=str, help="Directory for sub-experiments")
    parser.add_argument(
        "--run",
        default=None,
        type=str,
        help="Name of the python module that describes the run, e.g. pbt_baseline.experiments.humanoid_bench_pbt. "
        "Run module must be importable in your Python environment. It must define a global variable RUN_DESCRIPTION.",
    )
    parser.add_argument(
        "--backend",
        default="processes",
        choices=["processes", "slurm"],
        help="Launcher backend, use OS multiprocessing by default",
    )
    parser.add_argument("--pause_between", default=1, type=int, help="Pause in seconds between processes")
    parser.add_argument(
        "--experiment_suffix", default="", type=str, help="Append this to the name of the experiment dir"
    )

    partial_cfg, _ = parser.parse_known_args(args)

    if partial_cfg.backend == "slurm":
        parser = add_slurm_args(parser)
    elif partial_cfg.backend == "processes":
        parser = add_os_parallelism_args(parser)
    else:
        raise ValueError(f"Unknown backend: {partial_cfg.backend}")

    return parser


def parse_args():
    """Parse command line arguments."""
    args = launcher_argparser(sys.argv[1:]).parse_args(sys.argv[1:])
    return args


def main():
    """Main launcher function."""
    launcher_cfg = parse_args()

    try:
        # Import the run module
        run_module = importlib.import_module(f"{launcher_cfg.run}")
    except ImportError as exc:
        print(f"Could not import the run module {exc}")
        return 1

    run_description = run_module.RUN_DESCRIPTION
    run_description.experiment_suffix = launcher_cfg.experiment_suffix

    # Create train directory
    os.makedirs(launcher_cfg.train_dir, exist_ok=True)

    # Run experiments based on backend
    if launcher_cfg.backend == "processes":
        success = run(run_description, launcher_cfg)
    elif launcher_cfg.backend == "slurm":
        success = run_slurm(run_description, launcher_cfg)
    else:
        print(f"Unknown backend: {launcher_cfg.backend}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())