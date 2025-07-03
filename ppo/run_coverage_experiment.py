#!/usr/bin/env python3
"""
Script to run PPO training experiments with varying numbers of environments
to compare state coverage as the number of environments increases.
"""

import os
import subprocess
import sys
import time
from typing import List

def run_ppo_training(num_envs: int, base_output_dir: str = "coverage_experiments") -> None:
    """
    Run PPO training with specified number of environments.
    
    Args:
        num_envs: Number of parallel environments
        base_output_dir: Base directory for experiment outputs
    """
    # Create output directory for this experiment
    output_dir = os.path.join(base_output_dir, f"envs_{num_envs}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define training arguments
    cmd = [
        "python", "-m", "ppo.train",
        "--env_name", "G1JoystickRoughTerrain",
        "--num_envs", str(num_envs),
        "--total_timesteps", "500000",  # 500k steps as requested
        "--output_dir", output_dir,
        "--exp_name", f"coverage_exp_envs_{num_envs}",
        "--seed", "42",  # Fixed seed for reproducibility
        "--eval_interval", "50000",  # Evaluate every 50k steps
        "--save_interval", "0",  # Disable intermediate saves to save space
        "--log_interval", "10000",  # Log every 10k steps
    ]
    
    print(f"{'='*60}")
    print(f"Starting PPO training with {num_envs} environments")
    print(f"Output directory: {output_dir}")
    print(f"Total timesteps: 500,000")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Run the training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✅ Training completed successfully for {num_envs} environments")
        print(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {num_envs} environments")
        print(f"Error: {e}")
        raise
    
    print(f"{'='*60}\n")

def run_coverage_experiments(env_counts: List[int]) -> None:
    """
    Run coverage experiments for different environment counts.
    
    Args:
        env_counts: List of environment counts to test
    """
    print("🚀 Starting PPO Coverage Experiments")
    print(f"Environment counts to test: {env_counts}")
    print(f"Total timesteps per experiment: 500,000")
    print(f"Number of experiments: {len(env_counts)}")
    
    total_start_time = time.time()
    
    for i, num_envs in enumerate(env_counts):
        print(f"\n📊 Experiment {i+1}/{len(env_counts)}: {num_envs} environments")
        
        try:
            run_ppo_training(num_envs)
        except Exception as e:
            print(f"❌ Experiment failed for {num_envs} environments: {e}")
            continue
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n🎉 All experiments completed!")
    print(f"Total time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)")
    print(f"Results saved in: coverage_experiments/")
    
    # Print summary of results
    print(f"\n📈 Coverage Results Summary:")
    print(f"{'='*60}")
    print(f"Environment Count | Coverage File")
    print(f"{'='*60}")
    
    for num_envs in env_counts:
        output_dir = f"coverage_experiments/envs_{num_envs}"
        # Look for coverage files
        try:
            files = os.listdir(output_dir)
            coverage_files = [f for f in files if f.endswith('.pkl') and 'coverage' in f]
            if coverage_files:
                print(f"{num_envs:>15} | {coverage_files[0]}")
            else:
                print(f"{num_envs:>15} | No coverage file found")
        except FileNotFoundError:
            print(f"{num_envs:>15} | Output directory not found")
    
    print(f"{'='*60}")

def analyze_coverage_results(env_counts: List[int]) -> None:
    """
    Analyze the coverage results from the experiments.
    
    Args:
        env_counts: List of environment counts that were tested
    """
    import pickle
    
    print(f"\n📊 Coverage Analysis")
    print(f"{'='*80}")
    print(f"{'Envs':<8} | {'Final Coverage':<15} | {'Occupied Cells':<15} | {'Data Points':<12} | {'Max Coverage':<15}")
    print(f"{'='*80}")
    
    results = []
    
    for num_envs in env_counts:
        output_dir = f"coverage_experiments/envs_{num_envs}"
        
        try:
            # Find coverage file
            files = os.listdir(output_dir)
            coverage_files = [f for f in files if f.endswith('.pkl') and 'coverage' in f]
            
            if not coverage_files:
                print(f"{num_envs:<8} | No coverage data found")
                continue
                
            coverage_file = os.path.join(output_dir, coverage_files[0])
            
            # Load coverage data
            with open(coverage_file, 'rb') as f:
                data = pickle.load(f)
            
            final_coverage = data['final_coverage']
            occupied_cells = data['summary']['occupied_cells']
            total_cells = data['summary']['total_cells']
            data_points = data['total_data_points']
            max_coverage = data['summary'].get('max_coverage', final_coverage)
            
            print(f"{num_envs:<8} | {final_coverage:<15.4f} | {occupied_cells}/{total_cells:<9} | {data_points:<12} | {max_coverage:<15.4f}")
            
            results.append({
                'num_envs': num_envs,
                'final_coverage': final_coverage,
                'occupied_cells': occupied_cells,
                'total_cells': total_cells,
                'data_points': data_points,
                'max_coverage': max_coverage
            })
            
        except Exception as e:
            print(f"{num_envs:<8} | Error loading data: {e}")
    
    print(f"{'='*80}")
    
    # Create a simple analysis plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        if results:
            env_counts_plot = [r['num_envs'] for r in results]
            final_coverages = [r['final_coverage'] for r in results]
            max_coverages = [r['max_coverage'] for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(env_counts_plot, final_coverages, 'bo-', label='Final Coverage')
            plt.plot(env_counts_plot, max_coverages, 'ro-', label='Max Coverage')
            plt.xlabel('Number of Environments')
            plt.ylabel('Coverage')
            plt.title('State Coverage vs Number of Environments')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            
            # Save plot
            plot_path = "coverage_experiments/coverage_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Coverage comparison plot saved to: {plot_path}")
            
    except ImportError:
        print("📊 Matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"📊 Error creating plot: {e}")

def main():
    """Main function to run the coverage experiments."""
    # Environment counts to test
    env_counts = [512, 1024, 2048, 4096]
    
    # Check if we should only run analysis
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-only":
        analyze_coverage_results(env_counts)
        return
    
    # Run the experiments
    try:
        run_coverage_experiments(env_counts)
        
        # Analyze results after all experiments complete
        print("\n" + "="*60)
        analyze_coverage_results(env_counts)
        
    except KeyboardInterrupt:
        print("\n❌ Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiments failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()