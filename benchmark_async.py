"""Benchmark script to compare synchronous vs asynchronous PPO training speed."""

import time
import torch
import numpy as np
import subprocess
import sys
import os
from dataclasses import dataclass
from typing import Dict, Any
import argparse


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    total_time: float
    steps_completed: int
    steps_per_second: float
    avg_reward: float
    final_policy_loss: float
    success: bool
    error_message: str = ""


def run_training_benchmark(
    training_script: str,
    steps: int = 5_000_000,
    env_name: str = "T1JoystickFlatTerrain",
    num_envs: int = 32768,
    rollout_length: int = 256,
    timeout: int = 3600,  # 1 hour timeout
) -> BenchmarkResult:
    """Run a single training benchmark and capture results."""

    print(f"\n{'='*60}")
    print(f"Running benchmark: {training_script}")
    print(f"Target steps: {steps:,}")
    print(f"Environments: {num_envs}")
    print(f"Rollout length: {rollout_length}")
    print(f"{'='*60}")

    # Prepare command
    cmd = [
        sys.executable,
        "-m",
        f"ppo.{training_script}",
        "--env_name",
        env_name,
        "--total_timesteps",
        str(steps),
        "--num_envs",
        str(num_envs),
        "--rollout_length",
        str(rollout_length),
        "--eval_interval",
        "0",  # Disable evaluation for pure speed test
        # "--use_wandb", "False",  # Disable wandb logging
        "--save_interval",
        "0",  # Disable saving
        "--log_interval",
        str(steps // 10),  # Log 10 times during run
        "--amp",  # Enable AMP for best performance
        "--compile",  # Enable torch.compile for best performance
    ]

    start_time = time.time()
    success = False
    steps_completed = 0
    avg_reward = 0.0
    final_policy_loss = 0.0
    error_message = ""

    try:
        print(f"Command: {' '.join(cmd)}")
        print("Starting training...")

        # Run the training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Monitor output and capture metrics
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                print(output.strip())

                # Parse key metrics from output
                if (
                    "Training Progress - Step" in output
                    or "Async GPU Training Progress - Step" in output
                ):
                    try:
                        # Extract steps completed
                        if "Step " in output:
                            step_part = (
                                output.split("Step ")[1].split("/")[0].replace(",", "")
                            )
                            steps_completed = int(step_part)
                    except:
                        pass

                if "Avg Reward:" in output:
                    try:
                        reward_part = output.split("Avg Reward: ")[1].split(" |")[0]
                        avg_reward = float(reward_part)
                    except:
                        pass

                if "Policy:" in output:
                    try:
                        policy_part = output.split("Policy: ")[1].split(" |")[0]
                        final_policy_loss = float(policy_part)
                    except:
                        pass

        # Wait for process to complete
        return_code = process.wait(timeout=timeout)

        if return_code == 0:
            success = True
            print(f"✓ Training completed successfully!")
        else:
            error_message = f"Process exited with code {return_code}"
            print(f"✗ Training failed with return code: {return_code}")

    except subprocess.TimeoutExpired:
        process.kill()
        error_message = f"Training timed out after {timeout} seconds"
        print(f"✗ Training timed out after {timeout} seconds")

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        print(f"✗ Unexpected error: {e}")

    total_time = time.time() - start_time
    steps_per_second = steps_completed / total_time if total_time > 0 else 0

    result = BenchmarkResult(
        name=training_script,
        total_time=total_time,
        steps_completed=steps_completed,
        steps_per_second=steps_per_second,
        avg_reward=avg_reward,
        final_policy_loss=final_policy_loss,
        success=success,
        error_message=error_message,
    )

    print(f"\nBenchmark Results for {training_script}:")
    print(f"  Success: {success}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Steps completed: {steps_completed:,}/{steps:,}")
    print(f"  Steps/second: {steps_per_second:.1f}")
    print(f"  Avg reward: {avg_reward:.3f}")
    print(f"  Final policy loss: {final_policy_loss:.6f}")
    if error_message:
        print(f"  Error: {error_message}")

    return result


def print_comparison(sync_result: BenchmarkResult, async_result: BenchmarkResult):
    """Print detailed comparison between sync and async results."""

    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Metric':<25} {'Synchronous':<20} {'Asynchronous':<20} {'Speedup':<15}")
    print(f"{'-'*80}")

    # Success status
    sync_status = "✓ Success" if sync_result.success else "✗ Failed"
    async_status = "✓ Success" if async_result.success else "✗ Failed"
    print(f"{'Status':<25} {sync_status:<20} {async_status:<20} {'-':<15}")

    # Only compare if both succeeded
    if sync_result.success and async_result.success:
        # Total time
        time_speedup = (
            sync_result.total_time / async_result.total_time
            if async_result.total_time > 0
            else 0
        )
        print(
            f"{'Total Time (s)':<25} {sync_result.total_time:<20.1f} {async_result.total_time:<20.1f} {time_speedup:<15.2f}x"
        )

        # Steps per second
        speed_speedup = (
            async_result.steps_per_second / sync_result.steps_per_second
            if sync_result.steps_per_second > 0
            else 0
        )
        print(
            f"{'Steps/Second':<25} {sync_result.steps_per_second:<20.1f} {async_result.steps_per_second:<20.1f} {speed_speedup:<15.2f}x"
        )

        # Steps completed
        print(
            f"{'Steps Completed':<25} {sync_result.steps_completed:<20,} {async_result.steps_completed:<20,} {'-':<15}"
        )

        # Training quality metrics
        print(
            f"{'Avg Reward':<25} {sync_result.avg_reward:<20.3f} {async_result.avg_reward:<20.3f} {'-':<15}"
        )
        print(
            f"{'Policy Loss':<25} {sync_result.final_policy_loss:<20.6f} {async_result.final_policy_loss:<20.6f} {'-':<15}"
        )

        # Efficiency analysis
        print(f"\n{'EFFICIENCY ANALYSIS':<80}")
        print(f"{'-'*80}")

        if time_speedup > 1.0:
            efficiency_gain = (time_speedup - 1.0) * 100
            print(f"🚀 Asynchronous training is {efficiency_gain:.1f}% faster!")

            # Estimate time savings for full training runs
            full_training_steps = 200_000_000  # Typical full training
            sync_full_time = (
                full_training_steps / sync_result.steps_per_second / 3600
            )  # hours
            async_full_time = (
                full_training_steps / async_result.steps_per_second / 3600
            )  # hours
            time_saved = sync_full_time - async_full_time

            print(f"📊 For a full 200M step training run:")
            print(f"   Synchronous:  {sync_full_time:.1f} hours")
            print(f"   Asynchronous: {async_full_time:.1f} hours")
            print(
                f"   Time saved:   {time_saved:.1f} hours ({time_saved*60:.0f} minutes)"
            )

        elif time_speedup < 1.0:
            efficiency_loss = (1.0 - time_speedup) * 100
            print(f"⚠️  Asynchronous training is {efficiency_loss:.1f}% slower!")
            print(
                "   This might indicate overhead from threading or suboptimal GPU utilization."
            )
        else:
            print("⚖️  Both implementations have similar performance.")

        # GPU utilization estimate
        theoretical_max_envs = 32768
        env_steps_per_sec_sync = sync_result.steps_per_second
        env_steps_per_sec_async = async_result.steps_per_second

        print(f"\n📈 GPU Utilization Analysis:")
        print(f"   Sync GPU throughput:  {env_steps_per_sec_sync:.0f} env-steps/sec")
        print(f"   Async GPU throughput: {env_steps_per_sec_async:.0f} env-steps/sec")
        print(
            f"   Improvement: {((env_steps_per_sec_async/env_steps_per_sec_sync - 1) * 100):+.1f}%"
        )

    else:
        print(f"\n⚠️  Cannot compare performance - one or both benchmarks failed:")
        if not sync_result.success:
            print(f"   Synchronous error: {sync_result.error_message}")
        if not async_result.success:
            print(f"   Asynchronous error: {async_result.error_message}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark synchronous vs asynchronous PPO training"
    )
    parser.add_argument(
        "--steps", type=int, default=50_000_000, help="Number of training steps"
    )
    parser.add_argument(
        "--env_name", type=str, default="T1JoystickFlatTerrain", help="Environment name"
    )
    parser.add_argument(
        "--num_envs", type=int, default=32768, help="Number of parallel environments"
    )
    parser.add_argument(
        "--rollout_length", type=int, default=256, help="Rollout length"
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument(
        "--skip_sync", action="store_true", help="Skip synchronous benchmark"
    )
    parser.add_argument(
        "--skip_async", action="store_true", help="Skip asynchronous benchmark"
    )

    args = parser.parse_args()

    print("🏁 Starting PPO Training Speed Benchmark")
    print(f"Target steps: {args.steps:,}")
    print(f"Environment: {args.env_name}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    results = {}

    # Run synchronous benchmark
    if not args.skip_sync:
        print("\n🔄 Running SYNCHRONOUS training benchmark...")
        sync_result = run_training_benchmark(
            "train",
            args.steps,
            args.env_name,
            args.num_envs,
            args.rollout_length,
            args.timeout,
        )
        results["sync"] = sync_result

    # Run asynchronous benchmark
    if not args.skip_async:
        print("\n⚡ Running ASYNCHRONOUS training benchmark...")
        async_result = run_training_benchmark(
            "async_train",
            args.steps,
            args.env_name,
            args.num_envs,
            args.rollout_length,
            args.timeout,
        )
        results["async"] = async_result

    # Compare results
    if "sync" in results and "async" in results:
        print_comparison(results["sync"], results["async"])
    elif "sync" in results:
        print(
            f"\n✓ Synchronous benchmark completed: {results['sync'].steps_per_second:.1f} steps/sec"
        )
    elif "async" in results:
        print(
            f"\n✓ Asynchronous benchmark completed: {results['async'].steps_per_second:.1f} steps/sec"
        )

    print("\n🏁 Benchmark completed!")


if __name__ == "__main__":
    main()
