#!/bin/bash

# Function to create and submit a job for a single environment
submit_job() {
    local env_name=$1
    local image_name=$2
    local job_name="joystick_${env_name}"
    
    # Create a temporary job script
    cat > "job_${env_name}.slurm" << EOF
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
module load singularity/4.3.1
module load cuda/12.6.3

# Test network connectivity before entering container
echo "=== HOST NETWORK TEST ==="
curl -I http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd
echo "=== HOST NETWORK TEST COMPLETE ==="

# Print environment variables and test network inside container
singularity exec -B /nfs/turbo/coe-mandmlab/bpatil:/nfs/turbo/coe-mandmlab/bpatil \
    --nv --overlay /nfs/turbo/coe-mandmlab/bpatil/containers/isaacsim/isaac_lab.img \
    /nfs/turbo/coe-mandmlab/bpatil/containers/isaacsim/isaac-lab-base.sif \
    /bin/bash -c "
        echo '=== CONTAINER ENVIRONMENT VARIABLES ==='
        env | sort
        echo '=== CONTAINER NETWORK TEST ==='
        curl -I http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd
        echo '=== CONTAINER NETWORK TEST COMPLETE ==='
        echo '=== PYTHON PATH AND USD RESOLUTION TEST ==='
        /isaac-sim/python.sh -c \"
import os
print('PYTHONPATH:', os.environ.get('PYTHONPATH', 'Not set'))
print('USD-related env vars:')
for k, v in os.environ.items():
    if 'USD' in k or 'OMNI' in k or 'ISAAC' in k or 'CARB' in k:
        print(f'{k}: {v}')
print('\\n=== Testing USD Stage Resolution ===')
try:
    from pxr import Usd
    stage = Usd.Stage.GetCurrent()
    if stage is None:
        stage = Usd.Stage.CreateInMemory()
    usd_path = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd'
    resolved = stage.ResolveIdentifierToEditTarget(usd_path)
    print(f'USD Resolution for {usd_path}: {resolved}')
except Exception as e:
    print(f'USD Resolution test failed: {e}')
\"
        echo '=== STARTING ACTUAL TRAINING ==='
        /isaac-sim/python.sh /nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/fast_td3/train.py \
            --env_name ${env_name} \
            --exp_name ${env_name} \
            --render_interval 0 \
            --seed 0
    "

# Create output directory
mkdir -p outputs/\${SLURM_JOB_ID}

EOF

    # Make the script executable
    chmod +x "job_${env_name}.slurm"
    
    # Submit the job
    sbatch "job_${env_name}.slurm"
    
    # Clean up the temporary script
    rm "job_${env_name}.slurm"
}

# Submit jobs for all environments with the correct image name number
submit_job "Isaac-Velocity-Flat-G1-v0" "1"
# submit_job "Isaac-Velocity-Rough-G1-v0" "2"
# submit_job "Isaac-Repose-Cube-Allegro-Direct-v0" "3"
# submit_job "Isaac-Repose-Cube-Shadow-Direct-v0" "4"