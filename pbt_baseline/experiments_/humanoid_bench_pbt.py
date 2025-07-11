"""PBT experiment configuration for Humanoid Bench environments."""

from pbt_baseline.launcher.run_description import ParamGrid, RunDescription, Experiment
from pbt_baseline.experiments_.run_utils import version


_env = 'h1hand-reach-v0'
_name = f'{_env}_{version}'
_iterations = 100000
_pbt_num_policies = 4

_params = ParamGrid([
    ('pbt_policy_idx', list(range(_pbt_num_policies))),
])

_wandb_activate = True
_wandb_group = f'pbt_{_name}'
_wandb_entity = 'your_wandb_entity'  # Replace with your wandb entity
_wandb_project = 'pbt_humanoid_bench'

_experiments = [
    Experiment(
        f'{_name}',
        f'python -m pbt_baseline.train '
        f'--env-name {_env} '
        f'--total-timesteps {_iterations} '
        f'--num-envs 4 '
        f'--seed 0 '
        f'--save-interval 20000 '
        f'--eval-interval 10000 '
        # f'--use-wandb '
        f'--project {_wandb_project} '
        f'--pbt-enabled '
        f'--pbt-num-policies {_pbt_num_policies} '
        f'--pbt-workspace workspace_{_name} '
        f'--pbt-initial-delay 10000 '
        f'--pbt-interval-steps 25000 '
        f'--pbt-start-after 10000',
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