#!/bin/bash

# Function to create and submit a job for a single environment
submit_job() {
    local env_name=$1
    local buffer_size=$2
    local use_privileged_buffer=$3
    local random_initial_state=$4
    local job_name="joystick_${env_name}"
    
    # Create a temporary job script
    cat > "job_${env_name}.slurm" << 'EOF'
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --account=bucherb_owned1
#SBATCH --partition=spgpu2
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gpu_cmode=shared
#SBATCH --exclude=gl1710

# Load any required modules
module load cuda/12.6.3

# Load conda
# . /opt/conda/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda activate /scratch/bucherb_root/bucherb0/shared_data/envs/fasttd3_mjp

# Create output directory
mkdir -p outputs/${SLURM_JOB_ID}

cd /nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/
export PYTHONPATH=$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3
export PYTHONPATH=$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/fast_td3
EOF

    # Choose between three different command strings based on the arguments
    if [[ "$use_privileged_buffer" == "true" ]]; then
        # Command with privileged buffer
        cat >> "job_${env_name}.slurm" << EOF
# Run the training
python -m fast_td3.train \\
    --env_name ${env_name} \\
    --buffer_snapshot_interval 0 \\
    --output_dir output_single \\
    --num_envs 1024 \\
    --buffer_size ${buffer_size} \\
    --total_timesteps 100000 \\
    --eval_interval 5000 \\
    --project MJX_Multi \\
    --use_privileged_buffer \\
    --privileged_buffer_dir 'output_test_single' \\
    --privileged_buffer_max_samples 100000 \\
    --privileged_buffer_max_files 10 \\
    --privileged_buffer_priority_alpha 0.6 \\
    --privileged_buffer_reset_prob 0.5 \\
    --use_wandb --exp_name reset_buffer_10k_env_1024 --render_interval 5000 --seed 524
EOF
    elif [[ "$random_initial_state" == "true" ]]; then
        # Command with random initial state
        cat >> "job_${env_name}.slurm" << EOF
# Run the training
python -m fast_td3.train \\
    --env_name ${env_name} \\
    --buffer_snapshot_interval 0 \\
    --output_dir output_single \\
    --num_envs 1024 \\
    --buffer_size ${buffer_size} \\
    --total_timesteps 100000 \\
    --eval_interval 5000 \\
    --project MJX_Multi \\
    --random_initial_state \\
    --use_wandb --exp_name random_reset_buffer_10k_env_1024 --render_interval 5000 --seed 524
EOF
    else
        # Command without privileged buffer or random initial state
        cat >> "job_${env_name}.slurm" << EOF
# Run the training
python -m fast_td3.train \\
    --env_name ${env_name} \\
    --buffer_snapshot_interval 0 \\
    --output_dir output_single \\
    --num_envs 1024 \\
    --buffer_size ${buffer_size} \\
    --total_timesteps 100000 \\
    --eval_interval 5000 \\
    --project MJX_Multi \\
    --use_wandb --exp_name no_reset_buffer_10k_env_1024 --render_interval 5000 --seed 524
EOF
    fi

    # Make the script executable
    chmod +x "job_${env_name}.slurm"
    
    # Submit the job
    sbatch "job_${env_name}.slurm"
    
    # Clean up the temporary script
    rm "job_${env_name}.slurm"
}

# Submit jobs for all environments
# submit_job "G1JoystickFlatTerrain"
#submit_job "G1JoystickRoughTerrain"
submit_job "T1JoystickFlatTerrain" 10240 "false" "true"
# submit_job "T1JoystickFlatTerrain" 10240 "false"
# submit_job "T1JoystickFlatTerrain" "10240"
# submit_job "T1JoystickFlatTerrain" "50480"
# submit_job "T1JoystickRoughTerrain"