"""PBT experiment configuration for IsaacLab environments."""

from pbt_baseline.launcher.run_description import ParamGrid, RunDescription, Experiment
from pbt_baseline.experiments_.run_utils import version


_env = 'Isaac-Velocity-Flat-G1-v0'
_name = f'{_env}_{version}'
_iterations = 50000
_pbt_num_policies = 6

_params = ParamGrid([
    ('pbt_policy_idx', list(range(_pbt_num_policies))),
])

_wandb_activate = True
_wandb_group = f'pbt_{_name}'
_wandb_entity = 'your_wandb_entity'  # Replace with your wandb entity
_wandb_project = 'pbt_isaaclab'

_experiments = [
    Experiment(
        f'{_name}',
        f'python -m pbt_baseline.train '
        f'env_name={_env} '
        f'total_timesteps={_iterations} '
        f'num_envs=4096 '
        f'num_steps=24 '
        f'hidden_dim=512 '
        f'seed=-1 '
        f'save_interval=10000 '
        f'eval_interval=5000 '
        f'use_wandb={_wandb_activate} '
        f'project={_wandb_project} '
        f'pbt_enabled=True '
        f'pbt_num_policies={_pbt_num_policies} '
        f'pbt_workspace=workspace_{_name} '
        f'pbt_initial_delay=5000 '
        f'pbt_interval_steps=12500 '
        f'pbt_start_after=5000',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription(
    f'{_name}',
    experiments=_experiments, 
    experiment_arg_name='experiment', 
    experiment_dir_arg_name='output_dir',
    param_prefix='', 
    customize_experiment_name=False,
)