"""Simple test script for PBT baseline."""

import os
import sys
import torch

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from pbt_baseline.hyperparams import get_args
        print("✓ hyperparams imported successfully")
    except Exception as e:
        print(f"✗ Failed to import hyperparams: {e}")
        return False
    
    try:
        from pbt_baseline.mutation import mutate
        from pbt_baseline.hyperparams import PPO_MUTATION_CONFIG
        print("✓ mutation imported successfully")
    except Exception as e:
        print(f"✗ Failed to import mutation: {e}")
        return False
    
    try:
        from pbt_baseline.ppo import PPOAgent, ActorCritic
        print("✓ ppo imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ppo: {e}")
        return False
    
    try:
        from pbt_baseline.pbt import PbtObserver
        print("✓ pbt imported successfully")
    except Exception as e:
        print(f"✗ Failed to import pbt: {e}")
        return False
    
    return True


def test_mutation():
    """Test mutation functionality."""
    print("\nTesting mutation...")
    
    from pbt_baseline.mutation import mutate
    from pbt_baseline.hyperparams import PPO_MUTATION_CONFIG
    
    # Test parameters
    params = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_coeff': 0.2,
        'hidden_dim': 256,
    }
    
    try:
        mutated_params = mutate(params, PPO_MUTATION_CONFIG, 0.8, 1.1, 1.5)
        print(f"✓ Mutation successful: {mutated_params}")
        return True
    except Exception as e:
        print(f"✗ Mutation failed: {e}")
        return False


def test_ppo_creation():
    """Test PPO agent creation."""
    print("\nTesting PPO agent creation...")
    
    from pbt_baseline.ppo import PPOAgent
    from pbt_baseline.hyperparams import get_args
    
    try:
        # Create minimal args
        args = get_args()
        args.env_name = "test_env"
        args.num_envs = 4
        args.num_steps = 8
        args.hidden_dim = 64
        
        device = torch.device("cpu")
        agent = PPOAgent(args, n_obs=10, n_act=3, device=device)
        
        print("✓ PPO agent created successfully")
        return True
    except Exception as e:
        print(f"✗ PPO agent creation failed: {e}")
        return False


def test_launcher_imports():
    """Test launcher imports."""
    print("\nTesting launcher imports...")
    
    try:
        from pbt_baseline.launcher.run_description import ParamGrid, RunDescription, Experiment
        print("✓ run_description imported successfully")
    except Exception as e:
        print(f"✗ Failed to import run_description: {e}")
        return False
    
    try:
        from pbt_baseline.launcher.run_processes import ProcessManager
        print("✓ run_processes imported successfully")
    except Exception as e:
        print(f"✗ Failed to import run_processes: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("PBT Baseline Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_mutation,
        test_ppo_creation,
        test_launcher_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PBT baseline is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)