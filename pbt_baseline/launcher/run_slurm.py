"""Slurm launcher for PBT experiments."""

import os
import subprocess
import time
from pathlib import Path


def add_slurm_args(parser):
    """Add Slurm-specific arguments."""
    parser.add_argument("--slurm_workdir", default=None, type=str, help="Slurm working directory")
    parser.add_argument("--slurm_gpus_per_job", default=1, type=int, help="GPUs per Slurm job")
    parser.add_argument("--slurm_cpus_per_gpu", default=8, type=int, help="CPUs per GPU")
    parser.add_argument("--slurm_mem_per_gpu", default="32G", type=str, help="Memory per GPU")
    parser.add_argument("--slurm_partition", default="gpu", type=str, help="Slurm partition")
    parser.add_argument("--slurm_time", default="24:00:00", type=str, help="Time limit")
    parser.add_argument("--slurm_account", default=None, type=str, help="Slurm account")
    parser.add_argument("--slurm_exclude", default=None, type=str, help="Nodes to exclude")
    parser.add_argument("--slurm_constraint", default=None, type=str, help="Node constraints")
    parser.add_argument("--slurm_array_parallelism", default=8, type=int, help="Max array job parallelism")
    return parser


def create_slurm_script(
    experiment_cmd: str,
    job_name: str,
    output_dir: str,
    launcher_cfg,
    array_size: int = 1,
    array_id: int = 0,
):
    """Create Slurm batch script."""
    
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_dir}/slurm-%j.out",
        f"#SBATCH --error={output_dir}/slurm-%j.err",
        f"#SBATCH --gres=gpu:{launcher_cfg.slurm_gpus_per_job}",
        f"#SBATCH --cpus-per-gpu={launcher_cfg.slurm_cpus_per_gpu}",
        f"#SBATCH --mem-per-gpu={launcher_cfg.slurm_mem_per_gpu}",
        f"#SBATCH --partition={launcher_cfg.slurm_partition}",
        f"#SBATCH --time={launcher_cfg.slurm_time}",
    ]
    
    # Optional parameters
    if launcher_cfg.slurm_account:
        script_lines.append(f"#SBATCH --account={launcher_cfg.slurm_account}")
    
    if launcher_cfg.slurm_exclude:
        script_lines.append(f"#SBATCH --exclude={launcher_cfg.slurm_exclude}")
    
    if launcher_cfg.slurm_constraint:
        script_lines.append(f"#SBATCH --constraint={launcher_cfg.slurm_constraint}")
    
    if array_size > 1:
        script_lines.extend([
            f"#SBATCH --array=0-{array_size-1}%{launcher_cfg.slurm_array_parallelism}",
        ])
    
    # Working directory
    if launcher_cfg.slurm_workdir:
        script_lines.append(f"#SBATCH --chdir={launcher_cfg.slurm_workdir}")
    
    script_lines.extend([
        "",
        "# Load modules if needed",
        "# module load python cuda",
        "",
        "# Activate conda environment if needed", 
        "# conda activate pbt_env",
        "",
        "# Set environment variables",
        "export OMP_NUM_THREADS=1",
        'export MUJOCO_GL="egl"',
        "",
        "# Print job info",
        "echo \"Job ID: $SLURM_JOB_ID\"",
        "echo \"Node: $SLURMD_NODENAME\"",
        "echo \"GPU: $CUDA_VISIBLE_DEVICES\"",
        "",
    ])
    
    if array_size > 1:
        script_lines.extend([
            "# Array job handling",
            "echo \"Array task ID: $SLURM_ARRAY_TASK_ID\"",
            "",
            "# Select experiment based on array task ID",
            f"case $SLURM_ARRAY_TASK_ID in",
        ])
        
        # This would be filled in by the caller with experiment-specific commands
        script_lines.extend([
            f"    {array_id}) {experiment_cmd} ;;",
            "    *) echo \"Invalid array task ID: $SLURM_ARRAY_TASK_ID\"; exit 1 ;;",
            "esac",
        ])
    else:
        script_lines.extend([
            f"{experiment_cmd}",
        ])
    
    return "\n".join(script_lines)


def create_array_job_script(experiments, job_name: str, output_dir: str, launcher_cfg):
    """Create Slurm array job script for multiple experiments."""
    
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_dir}/slurm-%A_%a.out",
        f"#SBATCH --error={output_dir}/slurm-%A_%a.err",
        f"#SBATCH --gres=gpu:{launcher_cfg.slurm_gpus_per_job}",
        f"#SBATCH --cpus-per-gpu={launcher_cfg.slurm_cpus_per_gpu}",
        f"#SBATCH --mem-per-gpu={launcher_cfg.slurm_mem_per_gpu}",
        f"#SBATCH --partition={launcher_cfg.slurm_partition}",
        f"#SBATCH --time={launcher_cfg.slurm_time}",
        f"#SBATCH --array=0-{len(experiments)-1}%{launcher_cfg.slurm_array_parallelism}",
    ]
    
    # Optional parameters
    if launcher_cfg.slurm_account:
        script_lines.append(f"#SBATCH --account={launcher_cfg.slurm_account}")
    
    if launcher_cfg.slurm_exclude:
        script_lines.append(f"#SBATCH --exclude={launcher_cfg.slurm_exclude}")
    
    if launcher_cfg.slurm_constraint:
        script_lines.append(f"#SBATCH --constraint={launcher_cfg.slurm_constraint}")
    
    if launcher_cfg.slurm_workdir:
        script_lines.append(f"#SBATCH --chdir={launcher_cfg.slurm_workdir}")
    
    script_lines.extend([
        "",
        "# Load modules if needed",
        "# module load python cuda",
        "",
        "# Activate conda environment if needed",
        "# conda activate pbt_env", 
        "",
        "# Set environment variables",
        "export OMP_NUM_THREADS=1",
        'export MUJOCO_GL="egl"',
        "",
        "# Print job info",
        "echo \"Job ID: $SLURM_JOB_ID\"",
        "echo \"Array task ID: $SLURM_ARRAY_TASK_ID\"",
        "echo \"Node: $SLURMD_NODENAME\"",
        "echo \"GPU: $CUDA_VISIBLE_DEVICES\"",
        "",
        "# Select experiment based on array task ID",
        "case $SLURM_ARRAY_TASK_ID in",
    ])
    
    # Add case for each experiment
    for i, experiment in enumerate(experiments):
        script_lines.append(f"    {i}) cd {experiment['dir']} && {experiment['cmd']} ;;")
    
    script_lines.extend([
        "    *) echo \"Invalid array task ID: $SLURM_ARRAY_TASK_ID\"; exit 1 ;;",
        "esac",
    ])
    
    return "\n".join(script_lines)


def run_slurm(run_description, launcher_cfg):
    """Run experiments using Slurm."""
    print(f"Starting Slurm run: {run_description.run_name}")
    
    # Generate all experiment commands
    experiments = run_description.generate_experiments(launcher_cfg.train_dir)
    print(f"Generated {len(experiments)} experiments")
    
    # Create output directory
    output_dir = f"{launcher_cfg.train_dir}/slurm_logs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create array job script
    job_name = f"pbt_{run_description.run_name}"
    script_content = create_array_job_script(experiments, job_name, output_dir, launcher_cfg)
    
    # Write script to file
    script_path = f"{output_dir}/run_array.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created Slurm array job script: {script_path}")
    print(f"Number of array tasks: {len(experiments)}")
    
    # Submit job
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            cwd=output_dir
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"Submitted Slurm array job: {job_id}")
            print(f"Monitor with: squeue -j {job_id}")
            print(f"Cancel with: scancel {job_id}")
            return True
        else:
            print(f"Failed to submit job: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error submitting Slurm job: {e}")
        return False