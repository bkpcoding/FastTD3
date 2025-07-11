"""Run description system for PBT experiments."""

import copy
import itertools
from typing import Dict, List, Optional


class ParamGrid:
    """Parameter grid for experiment generation."""
    
    def __init__(self, grid_params):
        """
        Args:
            grid_params: List of tuples (param_name, param_values)
        """
        self.grid_params = grid_params
    
    def generate_params(self, randomize=True):
        """Generate all parameter combinations."""
        param_names = [param[0] for param in self.grid_params]
        param_values = [param[1] for param in self.grid_params]
        
        all_combinations = list(itertools.product(*param_values))
        
        if randomize:
            import random
            random.shuffle(all_combinations)
        
        return [dict(zip(param_names, combination)) for combination in all_combinations]


class Experiment:
    """Experiment configuration."""
    
    def __init__(self, name: str, cmd: str, param_combinations: List[Dict]):
        self.name = name
        self.cmd = cmd
        self.param_combinations = param_combinations


class RunDescription:
    """Complete description of a run with multiple experiments."""
    
    def __init__(
        self,
        run_name: str,
        experiments: List[Experiment],
        experiment_arg_name: str = "experiment",
        experiment_dir_arg_name: str = "train_dir",
        param_prefix: str = "",
        customize_experiment_name: bool = True,
    ):
        self.run_name = run_name
        self.experiments = experiments
        self.experiment_arg_name = experiment_arg_name
        self.experiment_dir_arg_name = experiment_dir_arg_name
        self.param_prefix = param_prefix
        self.customize_experiment_name = customize_experiment_name
        self.experiment_suffix = ""
    
    def generate_experiments(self, train_dir: str):
        """Generate all experiment commands."""
        all_commands = []
        
        for experiment in self.experiments:
            for i, param_combination in enumerate(experiment.param_combinations):
                # Create experiment name
                if self.customize_experiment_name:
                    exp_name = f"{experiment.name}_{i:04d}"
                else:
                    exp_name = experiment.name
                
                if self.experiment_suffix:
                    exp_name += f"_{self.experiment_suffix}"
                
                # Create experiment directory
                exp_dir = f"{train_dir}/{exp_name}"
                
                # Build command
                cmd = experiment.cmd
                
                # Add experiment directory
                if self.experiment_dir_arg_name:
                    cmd += f" --{self.experiment_dir_arg_name.replace('_', '-')} {exp_dir}"
                
                # Add parameters
                for param_name, param_value in param_combination.items():
                    if self.param_prefix:
                        param_name = f"{self.param_prefix}.{param_name}"
                    # Convert underscores to dashes for command line
                    param_name_cli = param_name.replace('_', '-')
                    cmd += f" --{param_name_cli} {param_value}"
                
                all_commands.append({
                    'cmd': cmd,
                    'name': exp_name,
                    'dir': exp_dir,
                    'params': param_combination,
                })
        
        return all_commands