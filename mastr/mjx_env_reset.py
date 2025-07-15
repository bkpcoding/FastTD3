#!/usr/bin/env python3
"""
Test script to verify state save/restore functionality for training environment.
Tests deterministic behavior by saving state, taking actions, restoring state, 
and repeating the same actions to verify identical end states.
"""

import numpy as np
import torch
from mujoco_playground_env import make_env


def test_train_env_state_save_restore():
    """Test state save/restore across different environment instances."""
    print("=== Testing RSLRLBraxWrapper with Cross-Environment State Transfer ===")
    
    env_name = "T1JoystickFlatTerrain"
    seed = 42
    
    # Step 1: Create first environment with 2 environments
    print("\n--- Step 1: Create first environment (2 envs) ---")
    train_env_1, _, _ = make_env(
        env_name=env_name,
        seed=seed,
        num_envs=2,
        num_eval_envs=1,
        device_rank=None
    )
    
    # Reset first environment
    obs_1 = train_env_1.reset()
    print(f"First environment observation shape: {obs_1.shape}")
    
    # Define sequence of actions
    action_sequence_2env = []
    action_sequence_4env = []
    num_steps = 5
    
    for i in range(num_steps):
        # Actions for 2 environments
        action_2env = torch.randn(2, train_env_1.num_actions)
        action_sequence_2env.append(action_2env)
        
        # Actions for 4 environments (extend the 2-env actions)
        action_4env = torch.randn(4, train_env_1.num_actions)
        action_4env[:2] = action_2env  # First 2 actions same as 2-env case
        action_sequence_4env.append(action_4env)
    
    # Take 3 steps in first environment
    current_obs_1 = obs_1
    for i in range(3):
        print(f"Step {i+1}: action shape = {action_sequence_2env[i].shape}")
        next_obs, reward, done, info = train_env_1.step(action_sequence_2env[i])
        print(f"  rewards: {reward.flatten()}")
        print(f"  dones: {done.flatten()}")
        current_obs_1 = next_obs
    
    # Save states from first environment
    print("Saving states from first environment after 3 steps...")
    saved_states_from_env1 = train_env_1.save_state()
    print(f"Saved {len(saved_states_from_env1)} states from first environment")
    import ipdb; ipdb.set_trace()
    # Continue with remaining 2 steps in first environment
    for i in range(3, 5):
        print(f"Step {i+1}: action shape = {action_sequence_2env[i].shape}")
        next_obs, reward, done, info = train_env_1.step(action_sequence_2env[i])
        print(f"  rewards: {reward.flatten()}")
        print(f"  dones: {done.flatten()}")
        current_obs_1 = next_obs
    
    # Save final states from first environment
    final_states_env1 = current_obs_1.clone()
    final_rewards_env1 = reward.clone()
    print(f"Final states from env1: obs sums = {final_states_env1.sum(dim=1)}")
    
    # Step 2: Create second environment with 4 environments
    print("\n--- Step 2: Create second environment (4 envs) ---")
    train_env_2, _, _ = make_env(
        env_name=env_name,
        seed=seed + 100,  # Different seed for different initialization
        num_envs=4,
        num_eval_envs=1,
        device_rank=None
    )
    
    # Reset second environment
    obs_2 = train_env_2.reset()
    print(f"Second environment observation shape: {obs_2.shape}")
    
    # Restore environments 1 and 3 in second environment with saved states from first environment
    print("\n--- Step 3: Restore states from env1 to env2 ---")
    env_mask = torch.tensor([True, True, False, False])  # Restore envs 0 and 1
    states_to_restore = [saved_states_from_env1[0], saved_states_from_env1[1]]  # Both states from env1

    print(f"Restoring environments {torch.where(env_mask)[0].tolist()} with states from first environment")
    train_env_2.restore_state(states_to_restore, env_mask)
    
    # Take the same remaining 2 steps in second environment
    print("\n--- Step 4: Take same actions in second environment ---")
    current_obs_2 = None
    for i in range(3, 5):
        print(f"Step {i+1}: action shape = {action_sequence_4env[i].shape}")
        next_obs, reward, done, info = train_env_2.step(action_sequence_4env[i])
        print(f"  rewards: {reward.flatten()}")
        print(f"  dones: {done.flatten()}")
        current_obs_2 = next_obs
    
    # Extract final states from environments 1 and 3 in second environment
    final_states_env2_restored = current_obs_2[[0, 1]].clone()
    final_rewards_env2_restored = reward[[0, 1]].clone()
    print(f"Final states from env2 (restored): obs sums = {final_states_env2_restored.sum(dim=1)}")
    
    # Step 5: Compare final states
    print("\n--- Step 5: Comparison ---")
    print("Comparing env1[0] final state with env2[1] final state:")
    state_diff_0_1 = torch.abs(final_states_env1[0] - final_states_env2_restored[0]).max().item()
    reward_diff_0_1 = torch.abs(final_rewards_env1[0] - final_rewards_env2_restored[0]).max().item()
    print(f"  State difference: {state_diff_0_1:.10f}")
    print(f"  Reward difference: {reward_diff_0_1:.10f}")
    
    print("Comparing env1[1] final state with env2[3] final state:")
    state_diff_1_3 = torch.abs(final_states_env1[1] - final_states_env2_restored[1]).max().item()
    reward_diff_1_3 = torch.abs(final_rewards_env1[1] - final_rewards_env2_restored[1]).max().item()
    print(f"  State difference: {state_diff_1_3:.10f}")
    print(f"  Reward difference: {reward_diff_1_3:.10f}")
    
    # Check if all differences are within tolerance
    max_state_diff = max(state_diff_0_1, state_diff_1_3)
    max_reward_diff = max(reward_diff_0_1, reward_diff_1_3)
    
    print(f"\nMax state difference: {max_state_diff:.10f}")
    print(f"Max reward difference: {max_reward_diff:.10f}")
    
    if max_state_diff < 1e-6 and max_reward_diff < 1e-6:
        print("✅ SUCCESS: Cross-environment state transfer works correctly!")
        return True
    else:
        print("❌ FAILURE: Cross-environment state transfer failed!")
        return False


if __name__ == "__main__":
    print("Testing state save/restore functionality for training environment...\n")
    
    # Test training environment  
    train_success = test_train_env_state_save_restore()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Training environment: {'✅ PASS' if train_success else '❌ FAIL'}")
    
    if train_success:
        print("\n🎉 Test passed! State save/restore is working correctly.")
    else:
        print("\n⚠️  Test failed. Check the implementation.")