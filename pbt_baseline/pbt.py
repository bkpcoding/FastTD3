"""Population-Based Training implementation adapted from isaacgymenvs."""

import logging
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

# Setup logger for PBT module
logger = logging.getLogger(__name__)


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
            logger.warning(f"Policy {policy_idx}: Exception {exc} in wandb.run.finish()")
            return
    
    logger.info(f"Policy {policy_idx}: Restarting self with args {modified_args}")
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
        logger.info("PBT job restarted from checkpoint, keep going...")
        return
    
    logger.info("PBT run without 'pbt_restart=True' - must be the very start of the experiment!")
    logger.info("Mutating initial set of hyperparameters!")
    
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
        
        # Evaluation tracking for PBT objective
        self.current_eval_reward = None
        
        # Cleanup tracking
        self.cleanup_interval = getattr(args, 'pbt_cleanup_interval', 5)
        self.cleanup_enabled = getattr(args, 'pbt_cleanup_enabled', True)
        self.last_cleanup_iteration = -1
    
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
    
    def process_eval_reward(self, eval_reward):
        """Process evaluation reward for PBT objective tracking."""
        if eval_reward is None:
            raise ValueError("Evaluation reward cannot be None. Evaluation must be performed before PBT step.")
        
        self.current_eval_reward = eval_reward
        self.target_objective_known = True
        self.curr_target_objective_value = float(eval_reward)
        
        if (
            self.best_objective_curr_iteration is None
            or self.curr_target_objective_value > self.best_objective_curr_iteration
        ):
            logger.info(
                f"Policy {self.policy_idx}: New best objective value "
                f"{self.curr_target_objective_value} in iteration {self.pbt_iteration}"
            )
            self.best_objective_curr_iteration = self.curr_target_objective_value
    
    def should_perform_pbt_step(self, global_step):
        """Check if PBT step should be performed."""
        if self.pbt_iteration == -1:
            self.pbt_iteration = global_step // self.pbt_params.interval_steps
            self.initial_env_frames = global_step
            logger.info(
                f"Policy {self.policy_idx}: PBT init. Steps: {global_step}, "
                f"pbt_iteration: {self.pbt_iteration}"
            )
        
        iteration = global_step // self.pbt_params.interval_steps
        
        if iteration <= self.pbt_iteration:
            return False
        
        if not self.target_objective_known:
            logger.debug(
                f"Policy {self.policy_idx}: Not enough episodes for objective estimation"
            )
            return False
        
        # Check if enough time has passed since experiment start
        sec_since_experiment_start = time.time() - self.experiment_start
        pbt_start_after_sec = 1 if self.pbt_params.dbg_mode else 30
        if sec_since_experiment_start < pbt_start_after_sec:
            logger.debug(
                f"Policy {self.policy_idx}: Not enough time passed since experiment start "
                f"{sec_since_experiment_start}"
            )
            return False
        
        return True
    
    def perform_pbt_step(self, global_step, policy, experiment_name):
        """Perform PBT step: save checkpoint and potentially replace policy."""
        logger.info(f"Policy {self.policy_idx}: New pbt iteration!")
        self.pbt_iteration = global_step // self.pbt_params.interval_steps
        
        try:
            self._save_pbt_checkpoint(policy, global_step, experiment_name)
        except Exception as exc:
            logger.error(f"Policy {self.policy_idx}: Exception {exc} when saving PBT checkpoint!")
            return
        
        try:
            checkpoints = self._load_population_checkpoints()
        except Exception as exc:
            logger.error(f"Policy {self.policy_idx}: Exception {exc} when loading checkpoints!")
            return
        
        try:
            self._cleanup(checkpoints)
        except Exception as exc:
            logger.warning(f"Policy {self.policy_idx}: Exception {exc} during cleanup!")
        
        # Cleanup bottom performers based on PBT logic
        if (self.cleanup_enabled and self.cleanup_interval > 0 and 
            self.pbt_iteration - self.last_cleanup_iteration >= self.cleanup_interval):
            try:
                logger.info(f"Policy {self.policy_idx}: Performing PBT-based cleanup...")
                self.cleanup_bottom_performers(checkpoints)
                self.last_cleanup_iteration = self.pbt_iteration
            except Exception as exc:
                logger.warning(f"Policy {self.policy_idx}: Exception {exc} during PBT cleanup!")
        
        logger.debug(f"Policy {self.policy_idx}: Checking if we should replace policy in {checkpoints}")
        # import ipdb; ipdb.set_trace()  # Commented out debug breakpoint
        # Determine if replacement should occur
        should_replace, replacement_policy = self._should_replace_policy(checkpoints)
        
        if should_replace:
            # Mark current policy for cleanup if it's being replaced (not self-mutation)
            if replacement_policy != self.policy_idx:
                self.mark_policy_for_replacement(self.policy_idx, "replaced_in_pbt")
            
            self._replace_policy(replacement_policy, checkpoints, experiment_name, global_step)
        
        # Clean up any policies marked for replacement during this iteration
        self.cleanup_marked_policies()
        
        # Reset for next iteration
        self.best_objective_curr_iteration = None
        self.target_objective_known = False
    
    def _save_pbt_checkpoint(self, policy, global_step, experiment_name):
        """Save current policy checkpoint with performance metrics."""
        checkpoint_name = _checkpnt_name(self.pbt_iteration)
        checkpoint_path = os.path.join(self.curr_policy_workspace_dir, checkpoint_name)
        
        # Save checkpoint data
        checkpoint_data = {
            "policy_idx": self.policy_idx,
            "iteration": self.pbt_iteration,
            "global_step": global_step,
            "true_objective": self.curr_target_objective_value,
            "params": self.pbt_params.mutable_params,
            "experiment_name": experiment_name,
            "timestamp": time.time(),
        }
        
        # Save policy weights
        model_checkpoint_name = _model_checkpnt_name(self.pbt_iteration)
        model_checkpoint_path = os.path.join(self.curr_policy_workspace_dir, model_checkpoint_name)
        
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'global_step': global_step,
        }, model_checkpoint_path)
        
        # Save metadata
        with open(checkpoint_path, 'w') as f:
            yaml.dump(checkpoint_data, f)
        
        logger.info(f"Policy {self.policy_idx}: Saved checkpoint {checkpoint_path} with objective {self.curr_target_objective_value}")
    
    def _load_population_checkpoints(self):
        """Load checkpoints from all policies in the population."""
        checkpoints = {}
        
        for policy_idx in range(self.pbt_num_policies):
            policy_workspace = self._policy_workspace_dir(policy_idx)
            
            if not os.path.exists(policy_workspace):
                checkpoints[policy_idx] = None
                continue
            
            # Find the latest checkpoint for this policy
            checkpoint_files = [f for f in os.listdir(policy_workspace) if f.endswith('.yaml')]
            if not checkpoint_files:
                checkpoints[policy_idx] = None
                continue
            
            # Get the most recent checkpoint
            latest_checkpoint = max(checkpoint_files)
            checkpoint_path = os.path.join(policy_workspace, latest_checkpoint)
            
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = yaml.safe_load(f)
                
                # Load model weights path
                iteration = checkpoint_data['iteration']
                model_checkpoint_name = _model_checkpnt_name(iteration)
                model_checkpoint_path = os.path.join(policy_workspace, model_checkpoint_name)
                
                checkpoint_data['model_path'] = model_checkpoint_path
                checkpoints[policy_idx] = checkpoint_data
                
            except Exception as e:
                logger.warning(f"Policy {self.policy_idx}: Failed to load checkpoint for policy {policy_idx}: {e}")
                checkpoints[policy_idx] = None
        
        return checkpoints
    
    def _should_replace_policy(self, checkpoints):
        """Determine if current policy should be replaced."""
        # Get valid checkpoints with objectives
        valid_checkpoints = {
            idx: ckpt for idx, ckpt in checkpoints.items() 
            if ckpt is not None and ckpt['true_objective'] != _UNINITIALIZED_VALUE
        }
        
        if len(valid_checkpoints) < 2:
            logger.debug(f"Policy {self.policy_idx}: Not enough valid checkpoints for comparison")
            return False, None
        
        # Sort policies by objective (descending)
        policies_by_objective = sorted(
            valid_checkpoints.items(), 
            key=lambda x: x[1]['true_objective'], 
            reverse=True
        )
        
        objectives = [ckpt['true_objective'] for _, ckpt in policies_by_objective]
        
        # Determine replacement thresholds
        replace_worst = max(1, int(self.pbt_params.replace_fraction_worst * self.pbt_num_policies))
        replace_best = max(1, int(self.pbt_params.replace_fraction_best * self.pbt_num_policies))
        
        worst_policies = [idx for idx, _ in policies_by_objective[-replace_worst:]]
        best_policies = [idx for idx, _ in policies_by_objective[:replace_best]]
        
        logger.debug(f"Policy {self.policy_idx}: Objectives: {objectives}")
        logger.debug(f"Policy {self.policy_idx}: Best policies: {best_policies}, Worst policies: {worst_policies}")
        
        # Check if current policy should be replaced
        if self.policy_idx not in worst_policies:
            logger.debug(f"Policy {self.policy_idx}: Not among worst policies, no replacement needed")
            return False, None
        
        # Check if we've had enough training time
        if not self._enough_training_time():
            logger.debug(f"Policy {self.policy_idx}: Not enough training time for replacement")
            return False, None
        
        # Select a random policy from the best performers
        import random
        replacement_policy = random.choice(best_policies)
        
        candidate_objective = valid_checkpoints[replacement_policy]['true_objective']
        current_objective = self.curr_target_objective_value
        
        # Check if improvement is significant enough
        objective_delta = candidate_objective - current_objective
        objectives_std = np.std(objectives)
        threshold = self.pbt_params.replace_threshold_frac_std * objectives_std
        
        if objective_delta > threshold:
            logger.info(f"Policy {self.policy_idx}: Will be replaced by policy {replacement_policy} "
                  f"(improvement: {objective_delta:.4f} > threshold: {threshold:.4f})")
            return True, replacement_policy
        else:
            logger.info(f"Policy {self.policy_idx}: Improvement not significant enough "
                  f"({objective_delta:.4f} <= {threshold:.4f}), mutating current policy")
            return True, self.policy_idx  # Replace with self (mutation only)
    
    def _enough_training_time(self):
        """Check if policy has had enough training time."""
        time_since_start = time.time() - self.experiment_start
        min_time = 1 if self.pbt_params.dbg_mode else 30
        return time_since_start >= min_time
    
    def _replace_policy(self, replacement_policy_idx, checkpoints, experiment_name, global_step):
        """Replace current policy with another policy + mutation."""
        if replacement_policy_idx == self.policy_idx:
            # Just mutate current parameters
            new_params = mutate(
                self.pbt_params.mutable_params,
                self.pbt_params.params_to_mutate,
                self.pbt_params.mutation_rate,
                self.pbt_params.change_min,
                self.pbt_params.change_max,
            )
            checkpoint_to_load = None
        else:
            # Load parameters from replacement policy and mutate them
            replacement_ckpt = checkpoints[replacement_policy_idx]
            base_params = replacement_ckpt['params']
            
            new_params = mutate(
                base_params,
                self.pbt_params.params_to_mutate,
                self.pbt_params.mutation_rate,
                self.pbt_params.change_min,
                self.pbt_params.change_max,
            )
            checkpoint_to_load = replacement_ckpt['model_path']
        
        logger.info(f"Policy {self.policy_idx}: Restarting with new parameters from policy {replacement_policy_idx}")
        
        # Restart process with new parameters and optionally new weights
        _restart_process_with_new_params(
            self.policy_idx, new_params, checkpoint_to_load, experiment_name, self.with_wandb
        )
    
    def _policy_workspace_dir(self, policy_idx):
        """Get workspace directory for a specific policy."""
        return os.path.join(self.pbt_workspace_dir, f"policy_{policy_idx:02d}")
    
    def _cleanup(self, checkpoints):
        """Clean up old checkpoints to save space."""
        # Keep only the latest 3 checkpoints per policy
        for policy_idx in range(self.pbt_num_policies):
            policy_workspace = self._policy_workspace_dir(policy_idx)
            if not os.path.exists(policy_workspace):
                continue
            
            # Get all checkpoint files
            checkpoint_files = [f for f in os.listdir(policy_workspace) if f.endswith('.yaml')]
            model_files = [f for f in os.listdir(policy_workspace) if f.endswith('.pth')]
            
            # Sort by iteration number and keep only latest 3
            if len(checkpoint_files) > 3:
                checkpoint_files.sort()
                for old_file in checkpoint_files[:-3]:
                    try:
                        os.remove(os.path.join(policy_workspace, old_file))
                        logger.debug(f"Removed old checkpoint: {old_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {old_file}: {e}")
            
            if len(model_files) > 3:
                model_files.sort()
                for old_file in model_files[:-3]:
                    try:
                        os.remove(os.path.join(policy_workspace, old_file))
                        logger.debug(f"Removed old model file: {old_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove model file {old_file}: {e}")
    
    def cleanup_replaced_policies(self, replaced_policies):
        """Clean up policies that have been replaced/mutated in PBT.
        
        Args:
            replaced_policies: List of policy indices that were replaced (bottom performers)
        """
        if not replaced_policies:
            return
            
        logger.info(f"Policy {self.policy_idx}: Cleaning up replaced policies: {replaced_policies}")
        
        # Clean up policy workspaces for replaced policies
        for policy_idx in replaced_policies:
            policy_workspace = self._policy_workspace_dir(policy_idx)
            if os.path.exists(policy_workspace):
                try:
                    shutil.rmtree(policy_workspace)
                    logger.info(f"Removed workspace for replaced policy {policy_idx}: {policy_workspace}")
                except OSError as e:
                    logger.warning(f"Failed to remove workspace for policy {policy_idx}: {e}")
        
        # Clean up model files for replaced policies
        models_dir = os.path.join(os.path.dirname(self.train_dir), "models")
        if os.path.exists(models_dir):
            self._cleanup_models_for_policies(models_dir, replaced_policies)
    
    def cleanup_bottom_performers(self, checkpoints):
        """Identify and clean up bottom performing policies based on PBT logic.
        
        Args:
            checkpoints: Current population checkpoints
        """
        # Get valid checkpoints with objectives
        valid_checkpoints = {
            idx: ckpt for idx, ckpt in checkpoints.items() 
            if ckpt is not None and ckpt['true_objective'] != _UNINITIALIZED_VALUE
        }
        
        if len(valid_checkpoints) < 2:
            logger.debug(f"Policy {self.policy_idx}: Not enough valid checkpoints for cleanup")
            return
        
        # Sort policies by objective (descending)
        policies_by_objective = sorted(
            valid_checkpoints.items(), 
            key=lambda x: x[1]['true_objective'], 
            reverse=True
        )
        
        # Determine bottom performers to remove
        replace_worst = max(1, int(self.pbt_params.replace_fraction_worst * self.pbt_num_policies))
        worst_policies = [idx for idx, _ in policies_by_objective[-replace_worst:]]
        
        # Only clean up policies that are significantly worse than the best
        objectives = [ckpt['true_objective'] for _, ckpt in policies_by_objective]
        objectives_std = np.std(objectives)
        best_objective = objectives[0]
        
        policies_to_cleanup = []
        for policy_idx in worst_policies:
            policy_objective = valid_checkpoints[policy_idx]['true_objective']
            objective_gap = best_objective - policy_objective
            
            # Only cleanup if performance gap is significant
            if objective_gap > self.pbt_params.replace_threshold_frac_std * objectives_std:
                policies_to_cleanup.append(policy_idx)
        
        if policies_to_cleanup:
            logger.info(f"Policy {self.policy_idx}: Identified bottom performers for cleanup: {policies_to_cleanup}")
            self.cleanup_replaced_policies(policies_to_cleanup)
    
    def _cleanup_models_for_policies(self, models_dir, policy_indices):
        """Clean up model files for specific policies."""
        try:
            for model_file in os.listdir(models_dir):
                if model_file.endswith('.pth'):
                    model_path = os.path.join(models_dir, model_file)
                    
                    # Check if model belongs to removed policy
                    if '__p' in model_file:
                        try:
                            policy_part = model_file.split('__p')[1]
                            policy_idx = int(policy_part.split('_')[0])
                            if policy_idx in policy_indices:
                                os.remove(model_path)
                                logger.info(f"Removed model file for policy {policy_idx}: {model_file}")
                        except (ValueError, IndexError, OSError) as e:
                            logger.warning(f"Error removing model file {model_file}: {e}")
                        
        except OSError as e:
            logger.warning(f"Error cleaning models directory: {e}")
    
    def mark_policy_for_replacement(self, policy_idx, reason="bottom_performer"):
        """Mark a policy for cleanup after replacement.
        
        Args:
            policy_idx: Index of policy to mark for cleanup
            reason: Reason for replacement (for logging)
        """
        logger.info(f"Policy {self.policy_idx}: Marking policy {policy_idx} for cleanup ({reason})")
        
        # Store in a list to cleanup after PBT step completes
        if not hasattr(self, '_policies_to_cleanup'):
            self._policies_to_cleanup = []
        
        if policy_idx not in self._policies_to_cleanup:
            self._policies_to_cleanup.append(policy_idx)
    
    def cleanup_marked_policies(self):
        """Clean up policies that were marked for replacement during this PBT iteration."""
        if hasattr(self, '_policies_to_cleanup') and self._policies_to_cleanup:
            logger.info(f"Policy {self.policy_idx}: Cleaning up marked policies: {self._policies_to_cleanup}")
            self.cleanup_replaced_policies(self._policies_to_cleanup)
            self._policies_to_cleanup = []  # Clear the list