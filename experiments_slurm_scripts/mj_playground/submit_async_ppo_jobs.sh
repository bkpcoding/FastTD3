#!/bin/bash

# Function to create and submit a job for async PPO with different environment counts
submit_async_ppo_job() {
    local num_envs=$1
    local job_name="async_ppo_${num_envs}envs"
    
    # Create a temporary job script
    cat > "job_${num_envs}envs.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --account=bucherb_owned1
#SBATCH --partition=spgpu2
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gpu_cmode=shared
#SBATCH --exclude=gl1710

# Load required modules
module load cuda/12.6.3

# Load conda
conda init
source ~/.bashrc
conda activate /scratch/bucherb_root/bucherb0/shared_data/envs/fasttd3_mjp

# Create output directory
mkdir -p outputs/\${SLURM_JOB_ID}

cd /nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/
export PYTHONPATH=\$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3

# GPU optimization settings
export OMP_NUM_THREADS=1
export TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"

echo "==================== JOB INFO ===================="
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node: \${SLURMD_NODENAME}"
echo "Number of environments: ${num_envs}"
echo "Total timesteps: 200000000"
echo "Batch size: 1024"
echo "=================================================="

# Run async PPO training
python -m ppo.async_train \
    --env_name "T1JoystickFlatTerrain" \
    --total_timesteps 200000000 \
    --num_envs ${num_envs} \
    --batch_size 1024 \
    --rollout_length 64 \
    --learning_rate 3e-4 \
    --output_dir "outputs/\${SLURM_JOB_ID}" \
    --log_interval 50000 \
    --eval_interval 1000000 \
    --save_interval 10000000 \
    --num_eval_envs 10 \
    --compile \
    --amp \
    --use_wandb \
    --project "rl_scratch" \
    --exp_name "async_${num_envs}envs" \
    --seed $((1000 + num_envs))
EOF

    # Make the script executable
    chmod +x "job_${num_envs}envs.slurm"
    
    # Submit the job
    sbatch "job_${num_envs}envs.slurm"
    
    # Clean up the temporary script
    rm "job_${num_envs}envs.slurm"
}

# Submit jobs for all environment configurations
submit_async_ppo_job 1024
# submit_async_ppo_job 2048
submit_async_ppo_job 4096
# submit_async_ppo_job 8192

echo "All async PPO jobs submitted successfully!"