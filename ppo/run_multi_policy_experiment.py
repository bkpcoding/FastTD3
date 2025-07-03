#!/usr/bin/env python3
"""
Optimized script to run PPO training experiments with multiple independent policies.

This script compares different approaches to environment exploration:
1. Single policy with all environments vs multiple independent policies
2. Tracks COMBINED state space exploration across all policies (not individual averages)
3. Optimized for performance with parallel data loading and reduced logging

Each policy is initialized separately and trained independently without sharing
data or weights. The total environment count is split across policies:
- 1 policy: 512 environments
- 2 policies: 256 environments each  
- 4 policies: 128 environments each

Usage:
  python run_multi_policy_experiment.py           # Run all experiments
  python run_multi_policy_experiment.py --quick   # Test 1,2 policies only
  python run_multi_policy_experiment.py --analyze-only  # Analysis only
"""

import os
import subprocess
import sys
import time
import multiprocessing as mp
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_policy(policy_id: int, num_envs: int, num_policies: int, base_output_dir: str = "multi_policy_experiments") -> Tuple[int, bool, str]:
    """
    Run PPO training for a single policy.
    
    Args:
        policy_id: Unique identifier for this policy (0-indexed)
        num_envs: Number of environments for this policy
        num_policies: Total number of policies being trained
        base_output_dir: Base directory for experiment outputs
    
    Returns:
        Tuple of (policy_id, success, error_message)
    """
    # Create output directory for this policy
    exp_dir = os.path.join(base_output_dir, f"policies_{num_policies}")
    policy_dir = os.path.join(exp_dir, f"policy_{policy_id}")
    os.makedirs(policy_dir, exist_ok=True)
    
    # Each policy gets a different seed for independent initialization
    seed = 42 + policy_id * 1000
    
    # Define training arguments
    cmd = [
        "python", "-m", "ppo.train",
        "--env_name", "G1JoystickRoughTerrain",
        "--num_envs", str(num_envs),
        "--total_timesteps", "500000",  # 500k steps as in coverage experiment
        "--output_dir", policy_dir,
        "--exp_name", f"multi_policy_exp_{num_policies}policies_policy{policy_id}",
        "--seed", str(seed),
        "--eval_interval", "100000",  # Reduce eval frequency for speed
        "--save_interval", "0",  # Disable intermediate saves to save space
        "--log_interval", "50000",  # Reduce log frequency for speed
    ]
    
    print(f"🚀 Starting policy {policy_id+1}/{num_policies} with {num_envs} environments (seed={seed})")
    
    # Run the training
    start_time = time.time()
    try:
        # Redirect output to policy-specific log file
        log_file = os.path.join(policy_dir, f"training_log_policy_{policy_id}.txt")
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        success_msg = f"✅ Policy {policy_id+1} completed successfully in {training_time:.1f}s ({training_time/60:.1f}min)"
        print(success_msg)
        return (policy_id, True, success_msg)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ Policy {policy_id+1} failed: {e}"
        print(error_msg)
        return (policy_id, False, error_msg)


def run_multi_policy_experiment(num_policies: int, base_output_dir: str = "multi_policy_experiments") -> None:
    """
    Run experiment with multiple independent policies.
    Optimized for better resource utilization and faster execution.
    
    Args:
        num_policies: Number of independent policies to train (1, 2, or 4)
        base_output_dir: Base directory for experiment outputs
    """
    # Calculate environments per policy
    total_envs = 512
    envs_per_policy = total_envs // num_policies
    
    print(f"{'='*80}")
    print(f"🎯 Multi-Policy Experiment: {num_policies} Independent Policies")
    print(f"Total environments: {total_envs}")
    print(f"Environments per policy: {envs_per_policy}")
    print(f"Total timesteps per policy: 500,000")
    print(f"{'='*80}")
    
    # Create base experiment directory
    exp_dir = os.path.join(base_output_dir, f"policies_{num_policies}")
    os.makedirs(exp_dir, exist_ok=True)
    
    experiment_start_time = time.time()
    
    # Optimize worker count based on system resources and policy count
    # For GPU-intensive training, limit parallel jobs to avoid resource contention
    max_workers = min(num_policies, max(1, mp.cpu_count() // 4))  # Conservative allocation
    print(f"💻 Using {max_workers} parallel workers for {num_policies} policies")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all policy training jobs
        futures = []
        for policy_id in range(num_policies):
            future = executor.submit(run_single_policy, policy_id, envs_per_policy, num_policies, base_output_dir)
            futures.append(future)
        
        # Collect results as they complete with progress tracking
        results = []
        completed = 0
        for future in as_completed(futures):
            policy_id, success, message = future.result()
            results.append((policy_id, success, message))
            completed += 1
            print(f"🏁 Progress: {completed}/{num_policies} policies completed")
    
    experiment_end_time = time.time()
    total_time = experiment_end_time - experiment_start_time
    
    # Sort results by policy_id for consistent reporting
    results.sort(key=lambda x: x[0])
    
    # Report results
    print(f"\n{'='*80}")
    print(f"📊 Experiment Results: {num_policies} Policies")
    print(f"{'='*80}")
    
    successful_policies = 0
    for policy_id, success, message in results:
        print(f"Policy {policy_id+1}: {'✅ SUCCESS' if success else '❌ FAILED'}")
        if success:
            successful_policies += 1
    
    print(f"\n📈 Summary:")
    print(f"Successful policies: {successful_policies}/{num_policies}")
    print(f"Total experiment time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    print(f"Results saved in: {exp_dir}/")
    print(f"{'='*80}")


def load_policy_data(policy_path_info: Tuple[int, int, str]) -> dict:
    """
    Load coverage data for a single policy (helper function for parallel processing).
    
    Args:
        policy_path_info: Tuple of (num_policies, policy_id, policy_dir)
        
    Returns:
        Dictionary with policy data or None if failed
    """
    import pickle
    import numpy as np
    
    num_policies, policy_id, policy_dir = policy_path_info
    
    try:
        if not os.path.exists(policy_dir):
            return None
        
        files = os.listdir(policy_dir)
        coverage_files = [f for f in files if f.endswith('.pkl') and 'coverage' in f]
        
        if not coverage_files:
            return None
            
        coverage_file = os.path.join(policy_dir, coverage_files[0])
        
        with open(coverage_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert occupied cells to tuples for set operations
        policy_cells = set(tuple(cell) if isinstance(cell, (list, np.ndarray)) else cell 
                         for cell in data['occupied_cells'])
        
        return {
            'num_policies': num_policies,
            'policy_id': policy_id,
            'occupied_cells': policy_cells,
            'final_coverage': data['final_coverage'],
            'total_data_points': data['total_data_points'],
            'grid_size': data['grid_size']
        }
        
    except Exception as e:
        print(f"Warning: Error loading policy {policy_id} in {num_policies}-policy exp: {e}")
        return None


def analyze_multi_policy_results(policy_counts: List[int], base_output_dir: str = "multi_policy_experiments") -> None:
    """
    Analyze the results from multi-policy experiments with combined coverage tracking.
    Optimized with parallel data loading for better performance.
    
    Args:
        policy_counts: List of policy counts that were tested (e.g., [1, 2, 4])
        base_output_dir: Base directory containing experiment results
    """
    import pickle
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    print(f"\n📊 Multi-Policy Experiment Analysis (Combined Coverage)")
    print(f"{'='*100}")
    print(f"{'Policies':<8} | {'Combined Coverage':<18} | {'Individual Avg':<16} | {'Total Cells':<12} | {'Data Points':<12}")
    print(f"{'='*100}")
    
    all_results = {}
    
    # Prepare all policy paths for parallel loading
    policy_paths = []
    for num_policies in policy_counts:
        exp_dir = os.path.join(base_output_dir, f"policies_{num_policies}")
        for policy_id in range(num_policies):
            policy_dir = os.path.join(exp_dir, f"policy_{policy_id}")
            policy_paths.append((num_policies, policy_id, policy_dir))
    
    # Load all policy data in parallel
    print(f"Loading {len(policy_paths)} policy datasets in parallel...")
    
    with ProcessPoolExecutor(max_workers=min(8, len(policy_paths))) as executor:
        future_to_path = {executor.submit(load_policy_data, path_info): path_info 
                         for path_info in policy_paths}
        
        policy_data = {}
        for future in as_completed(future_to_path):
            result = future.result()
            if result:
                num_policies = result['num_policies']
                if num_policies not in policy_data:
                    policy_data[num_policies] = []
                policy_data[num_policies].append(result)
    
    # Process results for each policy count
    for num_policies in policy_counts:
        envs_per_policy = 512 // num_policies
        
        if num_policies not in policy_data:
            print(f"{num_policies:<8} | No valid data found")
            continue
        
        policies = policy_data[num_policies]
        if not policies:
            print(f"{num_policies:<8} | No valid data found")
            continue
        
        # Combine all occupied cells from all policies
        combined_occupied_cells = set()
        individual_coverages = []
        total_data_points = 0
        grid_size = policies[0]['grid_size']
        
        for policy in policies:
            combined_occupied_cells.update(policy['occupied_cells'])
            individual_coverages.append(policy['final_coverage'])
            total_data_points += policy['total_data_points']
        
        # Calculate metrics
        total_cells = grid_size ** 2
        combined_coverage = len(combined_occupied_cells) / total_cells
        individual_avg = np.mean(individual_coverages)
        
        print(f"{num_policies:<8} | {combined_coverage:<18.4f} | {individual_avg:<16.4f} | {len(combined_occupied_cells)}/{total_cells:<7} | {total_data_points:<12}")
        
        all_results[num_policies] = {
            'combined_coverage': combined_coverage,
            'individual_avg': individual_avg,
            'individual_coverages': individual_coverages,
            'combined_occupied_cells': len(combined_occupied_cells),
            'total_cells': total_cells,
            'total_data_points': total_data_points,
            'envs_per_policy': envs_per_policy
        }
    
    print(f"{'='*100}")
    
    # Enhanced analysis comparing combined vs individual coverage
    print(f"\n📈 Combined vs Individual Coverage Analysis")
    print(f"{'='*80}")
    print(f"{'Policies':<8} | {'Combined':<12} | {'Individual':<12} | {'Improvement':<12} | {'Efficiency':<12}")
    print(f"{'='*80}")
    
    summary_results = []
    
    for num_policies, results in all_results.items():
        combined_cov = results['combined_coverage']
        individual_avg = results['individual_avg']
        improvement = (combined_cov - individual_avg) / individual_avg * 100 if individual_avg > 0 else 0
        
        # Coverage efficiency: combined coverage per policy
        efficiency = combined_cov / num_policies
        
        print(f"{num_policies:<8} | {combined_cov:<12.4f} | {individual_avg:<12.4f} | {improvement:<12.1f}% | {efficiency:<12.4f}")
        
        summary_results.append({
            'num_policies': num_policies,
            'combined_coverage': combined_cov,
            'individual_avg': individual_avg,
            'improvement_pct': improvement,
            'efficiency': efficiency,
            'envs_per_policy': results['envs_per_policy']
        })
    
    print(f"{'='*80}")
    
    # Create enhanced comparison plots
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        if summary_results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            policy_counts_plot = [r['num_policies'] for r in summary_results]
            combined_coverages = [r['combined_coverage'] for r in summary_results]
            individual_avgs = [r['individual_avg'] for r in summary_results]
            improvements = [r['improvement_pct'] for r in summary_results]
            efficiencies = [r['efficiency'] for r in summary_results]
            envs_per_policy = [r['envs_per_policy'] for r in summary_results]
            
            # Plot 1: Combined vs Individual Coverage
            ax1.plot(policy_counts_plot, combined_coverages, 'bo-', linewidth=2, markersize=8, label='Combined Coverage')
            ax1.plot(policy_counts_plot, individual_avgs, 'ro-', linewidth=2, markersize=8, label='Individual Average')
            ax1.set_xlabel('Number of Policies')
            ax1.set_ylabel('Coverage')
            ax1.set_title('Combined vs Individual Coverage')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(policy_counts_plot)
            
            # Plot 2: Coverage Improvement Percentage
            ax2.bar(policy_counts_plot, improvements, color='green', alpha=0.7)
            ax2.set_xlabel('Number of Policies')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Coverage Improvement: Combined vs Individual')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(policy_counts_plot)
            
            # Plot 3: Coverage Efficiency (per policy)
            ax3.plot(policy_counts_plot, efficiencies, 'go-', linewidth=2, markersize=8)
            ax3.set_xlabel('Number of Policies')
            ax3.set_ylabel('Coverage per Policy')
            ax3.set_title('Coverage Efficiency (Combined Coverage / # Policies)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(policy_counts_plot)
            
            # Plot 4: Coverage vs Environments per Policy
            ax4.plot(envs_per_policy, combined_coverages, 'bo-', linewidth=2, markersize=8, label='Combined')
            ax4.plot(envs_per_policy, individual_avgs, 'ro-', linewidth=2, markersize=8, label='Individual Avg')
            ax4.set_xlabel('Environments per Policy')
            ax4.set_ylabel('Coverage')
            ax4.set_title('Coverage vs Environment Allocation')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xscale('log')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(base_output_dir, "multi_policy_combined_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Enhanced multi-policy analysis plot saved to: {plot_path}")
            
    except ImportError:
        print("📊 Matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"📊 Error creating plot: {e}")


def main():
    """Main function to run the multi-policy experiments with optimized execution."""
    # Policy counts to test: 1, 2, and 4 independent policies
    policy_counts = [1, 2, 4]
    
    # Command line argument parsing
    if len(sys.argv) > 1:
        if sys.argv[1] == "--analyze-only":
            analyze_multi_policy_results(policy_counts)
            return
        elif sys.argv[1] == "--quick":
            # Quick test with reduced configurations
            policy_counts = [1, 2]
            print("📎 Quick mode enabled: testing 1 and 2 policies only")
    
    print("🚀 Starting Multi-Policy Independence Experiments (Optimized)")
    print(f"Policy configurations to test: {policy_counts}")
    print(f"Total environments distributed: 512")
    print(f"Environment distribution:")
    for num_policies in policy_counts:
        envs_per_policy = 512 // num_policies
        print(f"  {num_policies} policies: {envs_per_policy} environments each")
    
    total_start_time = time.time()
    successful_experiments = []
    
    # Run experiments for each policy count with better error handling
    for i, num_policies in enumerate(policy_counts):
        print(f"\n{'='*80}")
        print(f"🎯 Experiment {i+1}/{len(policy_counts)}: {num_policies} Independent Policies")
        print(f"{'='*80}")
        
        exp_start_time = time.time()
        try:
            run_multi_policy_experiment(num_policies)
            exp_end_time = time.time()
            exp_duration = exp_end_time - exp_start_time
            successful_experiments.append(num_policies)
            print(f"✅ Experiment {num_policies} policies completed in {exp_duration:.1f}s ({exp_duration/60:.1f}min)")
        except KeyboardInterrupt:
            print(f"\n❌ Experiment interrupted by user")
            break
        except Exception as e:
            print(f"❌ Experiment failed for {num_policies} policies: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n🎉 Multi-policy experiments completed!")
    print(f"Successful experiments: {len(successful_experiments)}/{len(policy_counts)}")
    print(f"Completed configurations: {successful_experiments}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)")
    print(f"Results saved in: multi_policy_experiments/")
    
    # Analyze results after all experiments complete
    if successful_experiments:
        try:
            print(f"\n{'='*80}")
            print("📈 Starting analysis of completed experiments...")
            analyze_multi_policy_results(successful_experiments)
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ No successful experiments to analyze")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiments failed: {e}")
        sys.exit(1)