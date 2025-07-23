#!/bin/bash

# Function to create and submit a job for layer normalization experiments
submit_layer_norm_job() {
    local env_name=$1
    local use_layer_norm=$2
    local seed=$3
    local job_name="ppo_ln_${env_name}_${use_layer_norm}"
    cat > "job_${env_name}_ln_${use_layer_norm}_${seed}.slurm" << EOF
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

# Run the training with layer normalization
python -m ppo.train \\
    --env_name ${env_name} \\
    --exp_name ppo_${env_name}_ln_${use_layer_norm} \\
    --seed ${seed} \\
    --num_envs 128 \\
    --use_wandb \\
    --use_layer_norm \\
    --output_dir outputs/\${SLURM_JOB_ID}
EOF

    # Make the script executable
    chmod +x "job_${env_name}_ln_${use_layer_norm}_${seed}.slurm"
    
    # Submit the job
    sbatch "job_${env_name}_ln_${use_layer_norm}_${seed}.slurm"
    
    # Clean up the temporary script
    rm "job_${env_name}_ln_${use_layer_norm}_${seed}.slurm"
}

# Test layer normalization on/off
env_name="h1hand-hurdle-v0"

# Layer normalization enabled experiments
submit_layer_norm_job "${env_name}" "true" "0"
# submit_layer_norm_job "${env_name}" "true" "123"
# submit_layer_norm_job "${env_name}" "true" "1024"
