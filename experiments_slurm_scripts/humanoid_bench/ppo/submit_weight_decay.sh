#!/bin/bash

# Function to create and submit a job for weight decay experiments
submit_weight_decay_job() {
    local env_name=$1
    local weight_decay=$2
    local seed=$3
    local job_name="ppo_wd_${env_name}_${weight_decay}"
    cat > "job_${env_name}_wd_${weight_decay}_${seed}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}_${seed}
#SBATCH --output=${job_name}_${seed}_%j.out
#SBATCH --error=${job_name}_${seed}_%j.err
#SBATCH --account=bucherb_owned1
#SBATCH --partition=spgpu2
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --gpu_cmode=shared
#SBATCH --exclude=gl1710

# Load any required modules
module load cuda/12.6.3

# Load conda
conda init
source ~/.bashrc
conda activate /scratch/bucherb_root/bucherb0/shared_data/envs/fasttd3_hb

# Create output directory
mkdir -p outputs/\${SLURM_JOB_ID}

cd /nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/
export PYTHONPATH=\$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3
export PYTHONPATH=\$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/fast_td3

# Run the training with weight decay
python -m ppo.train \\
    --env_name ${env_name} \\
    --exp_name ppo_${env_name}_wd_${weight_decay} \\
    --seed ${seed} \\
    --num_envs 128 \\
    --use_wandb \\
    --weight_decay ${weight_decay} \\
    --output_dir outputs/\${SLURM_JOB_ID}
EOF

    # Make the script executable
    chmod +x "job_${env_name}_wd_${weight_decay}_${seed}.slurm"
    
    # Submit the job
    sbatch "job_${env_name}_wd_${weight_decay}_${seed}.slurm"
    
    # Clean up the temporary script
    rm "job_${env_name}_wd_${weight_decay}_${seed}.slurm"
}

# Test two weight decay values: 1e-4 and 1e-3
env_name="h1hand-hurdle-v0"

# Weight decay 1e-4 experiments
submit_weight_decay_job "${env_name}" "1e-4" "0"
# submit_weight_decay_job "${env_name}" "1e-4" "123"
# submit_weight_decay_job "${env_name}" "1e-4" "1024"

# Weight decay 1e-3 experiments
submit_weight_decay_job "${env_name}" "1e-3" "0"
# submit_weight_decay_job "${env_name}" "1e-3" "123"
# submit_weight_decay_job "${env_name}" "1e-3" "1024"
