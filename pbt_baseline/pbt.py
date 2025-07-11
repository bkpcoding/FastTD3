"""Population-Based Training implementation adapted from isaacgymenvs."""

import math
import os
import random
import shutil
import sys
import time
from os.path import join
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from pbt_baseline.mutation import mutate
from pbt_baseline.hyperparams import PPO_MUTATION_CONFIG


# Value for target objective when it is not known
_UNINITIALIZED_VALUE = float(-1e9)


def _checkpnt_name(iteration):
    """Generate checkpoint filename for iteration."""
    return f"{iteration:06d}.yaml"


def _model_checkpnt_name(iteration):
    """Generate model checkpoint filename for iteration."""
    return f"{iteration:06d}.pth"


def _flatten_params(params: Dict, prefix="", separator=".") -> Dict:
    """Flatten nested parameter dictionary."""
    all_params = {}
    for key, value in params.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            all_params.update(_flatten_params(value, new_key, separator))
        else:
            all_params[new_key] = value
    return all_params


def _filter_params(params: Dict, params_to_mutate: Dict) -> Dict:
    """Filter parameters to only include those that can be mutated."""
    filtered_params = dict()
    for key, value in params.items():
        if key in params_to_mutate:
            if isinstance(value, str):
                try:
                    # Try to convert values such as "1e-4" to floats
                    float_value = float(value)
                    value = float_value
                except ValueError:
                    pass
            filtered_params[key] = value
    return filtered_params


class PbtParams:
    """Configuration parameters for PBT."""
    
    def __init__(self, args):
        self.replace_fraction_best = 0.25
        self.replace_fraction_worst = 0.25
        
        self.replace_threshold_frac_std = args.pbt_replace_threshold_frac_std
        self.replace_threshold_frac_absolute = args.pbt_replace_threshold_frac_absolute
        self.mutation_rate = args.pbt_mutation_rate
        self.change_min = args.pbt_change_min
        self.change_max = args.pbt_change_max
        
        self.env_name = args.env_name
        self.dbg_mode = args.pbt_dbg_mode
        
        self.policy_idx = args.pbt_policy_idx
        self.num_policies = args.pbt_num_policies
        
        self.num_envs = args.num_envs
        self.workspace = args.pbt_workspace
        
        self.interval_steps = args.pbt_interval_steps
        self.start_after_steps = args.pbt_start_after
        self.initial_delay_steps = args.pbt_initial_delay
        
        # Use PPO mutation configuration
        self.params_to_mutate = PPO_MUTATION_CONFIG
        
        # Get mutable parameters from args
        self.mutable_params = {
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_coeff': args.clip_coeff,
            'ent_coeff': args.ent_coeff,
            'vf_coeff': args.vf_coeff,
            'update_epochs': args.update_epochs,
            'hidden_dim': args.hidden_dim,
        }
        
        self.with_wandb = args.use_wandb


def _restart_process_with_new_params(
    policy_idx: int,
    new_params: Dict,
    restart_from_checkpoint: Optional[str],
    experiment_name: Optional[str],
    with_wandb: bool,
) -> None:
    """Restart the training process with new parameters."""
    cli_args = sys.argv
    
    modified_args = [cli_args[0]]  # Path to Python script
    for arg in cli_args[1:]:
        if "=" not in arg:
            modified_args.append(arg)
        else:
            arg_name, arg_value = arg.split("=", 1)
            if arg_name in new_params or arg_name in [
                "checkpoint_path",
                "pbt_restart",
            ]:
                # Skip this parameter, it will be added later
                continue
            modified_args.extend([f"--{arg_name.replace('_', '-')}", str(arg_value)])
    
    # Add PBT restart flag
    modified_args.extend(["--pbt-restart"])
    
    if restart_from_checkpoint is not None:
        modified_args.extend(["--checkpoint-path", restart_from_checkpoint])
    
    # Add all the new (possibly mutated) parameters
    for param, value in new_params.items():
        param_name = param.replace('_', '-')
        modified_args.extend([f"--{param_name}", str(value)])
    
    if with_wandb:
        try:
            import wandb
            wandb.run.finish()
        except Exception as exc:
            print(f"Policy {policy_idx}: Exception {exc} in wandb.run.finish()")
            return
    
    print(f"Policy {policy_idx}: Restarting self with args {modified_args}", flush=True)
    # Change the first argument from script path to module name
    if modified_args[0].endswith('train.py'):
        modified_args[0] = '-m'
        modified_args.insert(1, 'pbt_baseline.train')
    os.execv(sys.executable, ["python3"] + modified_args)


def initial_pbt_check(args):
    """Check if this is a PBT restart and mutate initial parameters if needed."""
    if not args.pbt_enabled:
        return
    
    if hasattr(args, "pbt_restart") and args.pbt_restart:
        print("PBT job restarted from checkpoint, keep going...")
        return
    
    print("PBT run without 'pbt_restart=True' - must be the very start of the experiment!")
    print("Mutating initial set of hyperparameters!")
    
    pbt_params = PbtParams(args)
    new_params = mutate(
        pbt_params.mutable_params,
        pbt_params.params_to_mutate,
        pbt_params.mutation_rate,
        pbt_params.change_min,
        pbt_params.change_max,
    )
    _restart_process_with_new_params(
        pbt_params.policy_idx, new_params, None, None, False
    )


class PbtObserver:
    """PBT Observer for managing population-based training."""
    
    def __init__(self, args, train_dir="./train_dir"):
        self.pbt_params: PbtParams = PbtParams(args)
        self.policy_idx: int = self.pbt_params.policy_idx
        self.num_envs: int = self.pbt_params.num_envs
        self.pbt_num_policies: int = self.pbt_params.num_policies
        
        self.train_dir = train_dir
        self.pbt_workspace_dir = None
        self.curr_policy_workspace_dir = None
        
        self.pbt_iteration = -1
        self.initial_env_frames = -1
        
        self.finished_agents = set()
        self.last_target_objectives = [_UNINITIALIZED_VALUE] * self.pbt_params.num_envs
        
        self.curr_target_objective_value: float = _UNINITIALIZED_VALUE
        self.target_objective_known = False
        
        self.best_objective_curr_iteration: Optional[float] = None
        self.experiment_start = time.time()
        self.with_wandb = self.pbt_params.with_wandb
        
        # Episode tracking for reward-based objectives
        self.episode_rewards = []
        self.episodes_to_track = 100  # Track last 100 episodes
    
    def after_init(self, train_dir=None):
        """Initialize PBT workspace directories."""
        if train_dir is not None:
            self.train_dir = train_dir
            
        self.pbt_workspace_dir = join(self.train_dir, self.pbt_params.workspace)
        self.curr_policy_workspace_dir = self._policy_workspace_dir(self.pbt_params.policy_idx)
        os.makedirs(self.curr_policy_workspace_dir, exist_ok=True)
    
    def process_episode_reward(self, reward):
        """Process episode reward for objective tracking."""
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) > self.episodes_to_track:
            self.episode_rewards.pop(0)
        
        # Update target objective if we have enough episodes
        if len(self.episode_rewards) >= min(self.episodes_to_track, self.num_envs):
            self.target_objective_known = True
            self.curr_target_objective_value = float(np.mean(self.episode_rewards))
            
            if (
                self.best_objective_curr_iteration is None
                or self.curr_target_objective_value > self.best_objective_curr_iteration
            ):
                print(
                    f"Policy {self.policy_idx}: New best objective value "
                    f"{self.curr_target_objective_value} in iteration {self.pbt_iteration}"
                )
                self.best_objective_curr_iteration = self.curr_target_objective_value
    
    def should_perform_pbt_step(self, global_step):
        """Check if PBT step should be performed."""
        if self.pbt_iteration == -1:
            self.pbt_iteration = global_step // self.pbt_params.interval_steps
            self.initial_env_frames = global_step
            print(
                f"Policy {self.policy_idx}: PBT init. Steps: {global_step}, "
                f"pbt_iteration: {self.pbt_iteration}"
            )
        
        iteration = global_step // self.pbt_params.interval_steps
        
        if iteration <= self.pbt_iteration:
            return False
        
        if not self.target_objective_known:
            print(
                f"Policy {self.policy_idx}: Not enough episodes for objective estimation"
            )
            return False
        
        # Check if enough time has passed since experiment start
        sec_since_experiment_start = time.time() - self.experiment_start
        pbt_start_after_sec = 1 if self.pbt_params.dbg_mode else 30
        if sec_since_experiment_start < pbt_start_after_sec:
            print(
                f"Policy {self.policy_idx}: Not enough time passed since experiment start "
                f"{sec_since_experiment_start}"
            )
            return False
        
        return True
    
    def perform_pbt_step(self, global_step, agent, experiment_name):
        """Perform PBT step: save checkpoint and potentially replace policy."""
        print(f"Policy {self.policy_idx}: New pbt iteration!")
        self.pbt_iteration = global_step // self.pbt_params.interval_steps
        
        try:
            self._save_pbt_checkpoint(agent, global_step, experiment_name)
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when saving PBT checkpoint!")
            return
        
        try:
            checkpoints = self._load_population_checkpoints()
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when loading checkpoints!")
            return
        
        try:
            self._cleanup(checkpoints)
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} during cleanup!")
        
        # Determine if replacement should occur
        should_replace, replacement_policy = self._should_replace_policy(checkpoints)
        
        if should_replace:
            self._replace_policy(replacement_policy, checkpoints, experiment_name, global_step)
        
        # Reset for next iteration
        self.best_objective_curr_iteration = None
        self.target_objective_known = False
    
    def _should_replace_policy(self, checkpoints):
        """Determine if current policy should be replaced."""
        policies = list(range(self.pbt_num_policies))
        target_objectives = []
        
        for p in policies:
            if checkpoints[p] is None:
                target_objectives.append(_UNINITIALIZED_VALUE)
            else:
                target_objectives.append(checkpoints[p]["true_objective"])
        
        policies_sorted = sorted(zip(target_objectives, policies), reverse=True)
        objectives = [objective for objective, p in policies_sorted]
        policies_sorted = [p for objective, p in policies_sorted]
        
        objectives_filtered = [o for o in objectives if o > _UNINITIALIZED_VALUE]
        
        if len(objectives_filtered) <= max(2, self.pbt_num_policies // 2) and not self.pbt_params.dbg_mode:
            print(f"Policy {self.policy_idx}: Not enough data to start PBT, {objectives_filtered}")
            return False, None
        
        replace_worst = math.ceil(self.pbt_params.replace_fraction_worst * self.pbt_num_policies)
        replace_best = math.ceil(self.pbt_params.replace_fraction_best * self.pbt_num_policies)
        
        best_policies = policies_sorted[:replace_best]
        worst_policies = policies_sorted[-replace_worst:]
        
        if self.policy_idx not in worst_policies and not self.pbt_params.dbg_mode:
            print(f"Policy {self.policy_idx} is doing well, not among worst_policies={worst_policies}")
            return False, None
        
        # Check if current policy performance justifies replacement
        if (
            self.best_objective_curr_iteration is not None 
            and not self.pbt_params.dbg_mode
            and self.best_objective_curr_iteration >= min(objectives[:replace_best])
        ):
            print(
                f"Policy {self.policy_idx}: best_objective={self.best_objective_curr_iteration} "
                f"is better than some top policies. Keep training."
            )
            return False, None
        
        # Select replacement policy
        replacement_policy_candidate = random.choice(best_policies)
        candidate_objective = checkpoints[replacement_policy_candidate]["true_objective"]
        objective_delta = candidate_objective - self.curr_target_objective_value
        
        # Calculate threshold for replacement
        num_outliers = int(math.floor(0.2 * len(objectives_filtered)))
        if len(objectives_filtered) > num_outliers:
            objectives_filtered_sorted = sorted(objectives_filtered)
            objectives_std = np.std(objectives_filtered_sorted[num_outliers:])
        else:
            objectives_std = np.std(objectives_filtered)
        
        objective_threshold = self.pbt_params.replace_threshold_frac_std * objectives_std
        absolute_threshold = self.pbt_params.replace_threshold_frac_absolute * abs(candidate_objective)
        
        if objective_delta > objective_threshold and objective_delta > absolute_threshold:
            print(f"Replacing policy {self.policy_idx} with {replacement_policy_candidate}")
            return True, replacement_policy_candidate
        else:
            print(f"Policy {self.policy_idx}: Performance difference not sufficient for replacement")
            return True, self.policy_idx  # Replace with self (mutate parameters only)
    
    def _replace_policy(self, replacement_policy, checkpoints, experiment_name, global_step):
        """Replace current policy with another policy and mutate parameters."""
        # Choose whether to copy parameters or keep current ones
        if random.random() < 0.5:
            new_params = checkpoints[replacement_policy]["params"]
        else:
            new_params = self.pbt_params.mutable_params
        
        # Mutate parameters
        new_params = mutate(
            new_params,
            self.pbt_params.params_to_mutate,
            self.pbt_params.mutation_rate,
            self.pbt_params.change_min,
            self.pbt_params.change_max,
        )
        
        try:
            restart_checkpoint = os.path.abspath(checkpoints[replacement_policy]["checkpoint"])
            
            # Copy checkpoint to temp directory
            checkpoint_tmp_dir = f"/tmp/{experiment_name}_p{self.policy_idx}"
            if os.path.isdir(checkpoint_tmp_dir):
                shutil.rmtree(checkpoint_tmp_dir)
            
            os.makedirs(checkpoint_tmp_dir, exist_ok=True)
            restart_checkpoint_tmp = join(checkpoint_tmp_dir, os.path.basename(restart_checkpoint))
            shutil.copyfile(restart_checkpoint, restart_checkpoint_tmp)
            
            # Rewrite checkpoint with current step
            self._rewrite_checkpoint(restart_checkpoint_tmp, global_step)
            
        except Exception as exc:
            print(f"Policy {self.policy_idx}: Exception {exc} when preparing checkpoint for restart")
            return
        
        print(f"Policy {self.policy_idx}: Restarting with mutated parameters!")
        _restart_process_with_new_params(
            self.policy_idx, new_params, restart_checkpoint_tmp, experiment_name, self.with_wandb
        )
    
    def _rewrite_checkpoint(self, restart_checkpoint_tmp: str, global_step: int) -> None:
        """Rewrite checkpoint with current global step."""
        state = torch.load(restart_checkpoint_tmp)
        print(f"Policy {self.policy_idx}: restarting from checkpoint, step {state.get('global_step', 'unknown')}")
        print(f"Replacing with step {global_step}...")
        state["global_step"] = global_step
        
        pbt_history = state.get("pbt_history", [])
        pbt_history.append((self.policy_idx, global_step, self.curr_target_objective_value))
        state["pbt_history"] = pbt_history
        
        torch.save(state, restart_checkpoint_tmp)
        print(f"Policy {self.policy_idx}: checkpoint rewritten!")
    
    def _save_pbt_checkpoint(self, agent, global_step, experiment_name):
        """Save PBT-specific checkpoint."""
        checkpoint_file = join(self.curr_policy_workspace_dir, _model_checkpnt_name(self.pbt_iteration))
        agent.save_checkpoint(checkpoint_file)
        
        pbt_checkpoint_file = join(self.curr_policy_workspace_dir, _checkpnt_name(self.pbt_iteration))
        
        pbt_checkpoint = {
            "iteration": self.pbt_iteration,
            "true_objective": self.curr_target_objective_value,
            "global_step": global_step,
            "params": self.pbt_params.mutable_params,
            "checkpoint": os.path.abspath(checkpoint_file),
            "pbt_checkpoint": os.path.abspath(pbt_checkpoint_file),
            "experiment_name": experiment_name,
        }
        
        with open(pbt_checkpoint_file, "w") as fobj:
            print(f"Policy {self.policy_idx}: Saving {pbt_checkpoint_file}...")
            yaml.dump(pbt_checkpoint, fobj)
    
    def _policy_workspace_dir(self, policy_idx):
        """Get workspace directory for a policy."""
        return join(self.pbt_workspace_dir, f"{policy_idx:03d}")
    
    def _load_population_checkpoints(self):
        """Load checkpoints for all policies in the population."""
        checkpoints = dict()
        
        for policy_idx in range(self.pbt_num_policies):
            checkpoints[policy_idx] = None
            
            policy_workspace_dir = self._policy_workspace_dir(policy_idx)
            
            if not os.path.isdir(policy_workspace_dir):
                continue
            
            pbt_checkpoint_files = [f for f in os.listdir(policy_workspace_dir) if f.endswith(".yaml")]
            pbt_checkpoint_files.sort(reverse=True)
            
            for pbt_checkpoint_file in pbt_checkpoint_files:
                iteration_str = pbt_checkpoint_file.split(".")[0]
                iteration = int(iteration_str)
                
                if iteration <= self.pbt_iteration:
                    with open(join(policy_workspace_dir, pbt_checkpoint_file), "r") as fobj:
                        print(f"Policy {self.policy_idx}: Loading policy-{policy_idx} {pbt_checkpoint_file}")
                        checkpoints[policy_idx] = yaml.load(fobj, Loader=yaml.FullLoader)
                        break
        
        return checkpoints
    
    def _cleanup(self, checkpoints):
        """Clean up old checkpoints."""
        iterations = []
        for policy_idx, checkpoint in checkpoints.items():
            if checkpoint is None:
                iterations.append(0)
            else:
                iterations.append(checkpoint["iteration"])
        
        oldest_iteration = sorted(iterations)[0]
        cleanup_threshold = oldest_iteration - 20
        
        pbt_checkpoint_files = [f for f in os.listdir(self.curr_policy_workspace_dir)]
        
        for f in pbt_checkpoint_files:
            if "." in f:
                iteration_idx = int(f.split(".")[0])
                if iteration_idx <= cleanup_threshold:
                    print(f"Policy {self.policy_idx}: PBT cleanup: removing checkpoint {f}")
                    try:
                        os.remove(join(self.curr_policy_workspace_dir, f))
                    except Exception:
                        pass  # Ignore cleanup errors