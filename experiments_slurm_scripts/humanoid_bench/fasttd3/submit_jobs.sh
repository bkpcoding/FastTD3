#!/bin/bash

# Function to create and submit a job for a single environment
submit_job() {
    local env_name=$1
    local job_name="joystick_${env_name}"
    local seed=$2
    cat > "job_${env_name}.slurm" << EOF
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
# . /opt/conda/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda activate /scratch/bucherb_root/bucherb0/shared_data/envs/fasttd3_hb

# Create output directory
mkdir -p outputs/\${SLURM_JOB_ID}

cd /nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/
export PYTHONPATH=\$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3
export PYTHONPATH=\$PYTHONPATH:/nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/fast_td3
# Run the training
python -m fast_td3.train \
    --env_name ${env_name} \
    --exp_name fasttd3_${env_name} \
    --seed ${seed} \
    --output_dir outputs/\${SLURM_JOB_ID}
EOF

    # Make the script executable
    chmod +x "job_${env_name}.slurm"
    
    # Submit the job
    sbatch "job_${env_name}.slurm"
    
    # Clean up the temporary script
    rm "job_${env_name}.slurm"
}

# Submit jobs for all environments
submit_job "h1hand-hurdle-v0" "0"
submit_job "h1hand-hurdle-v0" "123"
submit_job "h1hand-hurdle-v0" "1024"
# submit_job "G1JoystickRoughTerrain"
# submit_job "T1JoystickFlatTerrain"
# submit_job "T1JoystickRoughTerrain"