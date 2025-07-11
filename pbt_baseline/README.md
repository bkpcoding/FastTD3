# PBT Baseline

A Population-Based Training (PBT) baseline implementation for reinforcement learning environments, supporting IsaacLab, MuJoCo Playground, Humanoid Bench, and MTBench environments.

## Overview

This PBT baseline provides:
- **PPO algorithm** as the base RL method
- **Population-Based Training** for evolutionary hyperparameter optimization
- **Multi-environment support** compatible with FastTD3's environment ecosystem
- **Distributed training** via Slurm or local multiprocessing
- **Comprehensive logging** and evaluation

## Installation

The PBT baseline reuses FastTD3's environment dependencies. Ensure you have:
- FastTD3 dependencies installed
- Access to the desired environments (IsaacLab, MuJoCo Playground, etc.)
- Python 3.8+
- PyTorch
- Weights & Biases (optional, for logging)

## Quick Start

### 1. Test the Installation

```bash
cd /nfs/turbo/coe-mandmlab/bpatil/projects/FastTD3/pbt_baseline
python test_simple.py
```

### 2. Run a Single Training (No PBT)

```bash
python train.py \
    env_name=h1hand-reach-v0 \
    total_timesteps=50000 \
    num_envs=64 \
    pbt_enabled=False
```

### 3. Run PBT Experiment

#### Local Multiprocessing
```bash
python launcher/run.py \
    --run pbt_baseline.experiments.humanoid_bench_pbt \
    --backend processes \
    --max_parallel 4 \
    --train_dir ./pbt_results
```

#### Slurm Cluster
```bash
python launcher/run.py \
    --run pbt_baseline.experiments.isaaclab_pbt \
    --backend slurm \
    --slurm_partition gpu \
    --slurm_time 24:00:00 \
    --train_dir ./pbt_results
```

## Environment Support

### Humanoid Bench
- Environments: `h1hand-*`, `h1-*`
- Example: `h1hand-reach-v0`, `h1hand-balance-simple-v0`
- Configuration: `experiments/humanoid_bench_pbt.py`

### IsaacLab  
- Environments: `Isaac-*`
- Example: `Isaac-Velocity-Flat-G1-v0`, `Isaac-Lift-Cube-Franka-v0`
- Configuration: `experiments/isaaclab_pbt.py`

### MuJoCo Playground
- Environments: Various locomotion and manipulation tasks
- Example: `G1JoystickFlatTerrain`, `LeapCubeReorient`
- Configuration: `experiments/mujoco_playground_pbt.py`

### MTBench
- Environments: `MTBench-*`
- Example: `MTBench-meta-world-v2-mt10`
- Configuration: `experiments/mtbench_pbt.py`

## PBT Configuration

Key PBT parameters:

- `pbt_num_policies`: Population size (4-8 recommended)
- `pbt_interval_steps`: Steps between PBT operations
- `pbt_mutation_rate`: Probability of mutating each parameter (0.8 default)
- `pbt_replace_fraction_worst`: Fraction of worst policies to replace (0.25 default)

### Hyperparameter Mutations

The following hyperparameters are mutated:
- `learning_rate`: Actor-critic learning rate
- `gamma`: Discount factor
- `gae_lambda`: GAE lambda parameter
- `clip_coeff`: PPO clipping coefficient
- `ent_coeff`: Entropy regularization
- `vf_coeff`: Value function coefficient
- `update_epochs`: Number of PPO epochs
- `hidden_dim`: Network hidden dimension

## Experiment Structure

```
pbt_baseline/
├── train.py              # Main training script
├── ppo.py                # PPO algorithm implementation
├── pbt.py                # PBT population management
├── mutation.py           # Hyperparameter mutation functions
├── hyperparams.py        # Configuration and hyperparameters
├── env_utils.py          # Environment creation utilities
├── experiments/          # Pre-configured experiments
│   ├── humanoid_bench_pbt.py
│   ├── isaaclab_pbt.py
│   ├── mujoco_playground_pbt.py
│   └── mtbench_pbt.py
└── launcher/             # Distributed training launchers
    ├── run.py
    ├── run_processes.py
    └── run_slurm.py
```

## Creating Custom Experiments

1. Create a new experiment file in `experiments/`:

```python
from pbt_baseline.launcher.run_description import ParamGrid, RunDescription, Experiment

_env = 'your-environment-name'
_pbt_num_policies = 6

_params = ParamGrid([
    ('pbt_policy_idx', list(range(_pbt_num_policies))),
])

_experiments = [
    Experiment(
        f'{_env}_pbt',
        f'python -m pbt_baseline.train '
        f'env_name={_env} '
        f'total_timesteps=100000 '
        f'pbt_enabled=True '
        f'pbt_num_policies={_pbt_num_policies}',
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription(
    f'{_env}_pbt',
    experiments=_experiments,
    experiment_arg_name='experiment',
    experiment_dir_arg_name='output_dir',
)
```

2. Run the experiment:
```bash
python launcher/run.py --run your_module.your_experiment
```

## Monitoring and Results

- **Weights & Biases**: Set `use_wandb=True` and configure your entity/project
- **Checkpoints**: Saved to `output_dir` with PBT history
- **Logs**: Training metrics, evaluation results, and PBT operations

## Comparison with FastTD3

| Feature | PBT Baseline | FastTD3 |
|---------|-------------|---------|
| Algorithm | PPO | TD3 + Distributional RL |
| Training | Population-Based | Single Policy |
| Environments | All FastTD3 envs | All FastTD3 envs |
| Hyperparameter Tuning | Automatic (PBT) | Manual |
| Computational Cost | Higher (population) | Lower (single) |

## Troubleshooting

### Import Errors
- Ensure FastTD3 dependencies are installed
- Check that environment-specific packages are available

### PBT Process Restarts
- PBT automatically restarts processes with new hyperparameters
- Check workspace directory for checkpoint files
- Monitor process logs for restart messages

### Slurm Issues
- Verify Slurm configuration in experiment files
- Check cluster resource availability
- Adjust `slurm_array_parallelism` for job queue limits

## Performance Tips

1. **Population Size**: Start with 4-6 policies, increase for complex environments
2. **Mutation Rate**: 0.8 works well for most cases
3. **Interval Steps**: Balance between exploration and exploitation
4. **Resource Management**: Use appropriate number of environments per GPU
5. **Checkpointing**: Regular saves allow for robust restarts

## Citation

This PBT baseline is adapted from NVIDIA's IsaacGymEnvs PBT implementation and integrates with the FastTD3 environment ecosystem.