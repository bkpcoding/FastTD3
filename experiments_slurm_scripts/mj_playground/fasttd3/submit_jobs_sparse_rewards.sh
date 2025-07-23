#!/bin/bash

# Environment names - edit these directly to change which environments to train
env1="AcrobotSwingupSparse"
env2="CartpoleBalanceSparse"
env3="CartpoleSwingupSparse"

# Function to create and submit a job for three environments running in parallel
submit_multi_env_job() {
    local env1=$1
    local env2=$2
    local env3=$3
    local job_name="multi_env_${env1}_${env2}_${env3}"
    local seed=0
    
    # Create a temporary job script
    cat > "job_multi_env.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --account=bucherb_owned1
#SBATCH --partition=spgpu2
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH --gpu_cmode=shared
#SBATCH --exclude=gl1710

# Load any required modules
module load cuda/12.6.3

# Load conda
conda init
source ~/.bashrc
conda activate /scratch/bucherb_root/bucherb0/shared_data/envs/fasttd3_mjp

# Create output directory
mkdir -p outputs/\${SLURM_JOB_ID}

cd /nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/
export PYTHONPATH=\$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3
export PYTHONPATH=\$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/fast_td3

# Set seed value
SEED=${seed}

# Function to run training with error handling
run_training() {
    local env_name=\$1
    local env_index=\$2
    
    echo "Starting training for \${env_name} (Environment \${env_index}) with seed \${SEED}"
    
    python -m fast_td3.train \\
        --env_name \${env_name} \\
        --exp_name fasttd3_\${env_name}_multi_env \\
        --seed \${SEED} \\
        --use_wandb \\
        --project MJX \\
        --num_envs 1024 \\
        --output_dir outputs/\${SLURM_JOB_ID}/\${env_name} &
    
    local pid=\$!
    echo "Started \${env_name} with PID \${pid}"
    return \${pid}
}

# Start all three training processes in the background
echo "Starting parallel training for three environments..."

run_training "${env1}" "1"
pid1=\$!

echo "Waiting 30 seconds before starting next environment to avoid wandb conflicts..."
sleep 30

run_training "${env2}" "2"
pid2=\$!

echo "Waiting 30 seconds before starting final environment..."
sleep 30

run_training "${env3}" "3"
pid3=\$!

# Store PIDs for monitoring
echo "Environment processes started:"
echo "  ${env1}: PID \${pid1}"
echo "  ${env2}: PID \${pid2}"
echo "  ${env3}: PID \${pid3}"

# Wait for all processes to complete
echo "Waiting for all training processes to complete..."

wait \${pid1}
exit_code1=\$?
echo "${env1} completed with exit code \${exit_code1}"

wait \${pid2}
exit_code2=\$?
echo "${env2} completed with exit code \${exit_code2}"

wait \${pid3}
exit_code3=\$?
echo "${env3} completed with exit code \${exit_code3}"

# Report final status
echo "All training processes completed:"
echo "  ${env1}: exit code \${exit_code1}"
echo "  ${env2}: exit code \${exit_code2}"
echo "  ${env3}: exit code \${exit_code3}"

# Exit with non-zero code if any process failed
if [ \${exit_code1} -ne 0 ] || [ \${exit_code2} -ne 0 ] || [ \${exit_code3} -ne 0 ]; then
    echo "One or more training processes failed"
    exit 1
else
    echo "All training processes completed successfully"
    exit 0
fi
EOF

    # Make the script executable
    chmod +x "job_multi_env.slurm"
    
    # Submit the job
    echo "Submitting job for environments: ${env1}, ${env2}, ${env3}"
    sbatch "job_multi_env.slurm"
    
    # Clean up the temporary script
    rm "job_multi_env.slurm"
}
# Submit the multi-environment job
submit_multi_env_job "${env1}" "${env2}" "${env3}"

echo "Job submitted successfully!"
