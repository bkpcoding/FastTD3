"""FAISS-based state storage and nearest neighbor search for PPO environments."""

import time
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
from dataclasses import dataclass, asdict
import torch
import faiss
from fast_td3.environments.mujoco_playground_env import make_env
from .hyperparams import get_args

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"  # Enable triton gemm

import torch._dynamo

torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True


@dataclass
class StateInfo:
    """Information associated with a stored state."""
    state_vector: np.ndarray
    episode_id: int
    step_id: int
    reward: float
    done: bool
    timestamp: float
    metadata: Dict[str, Any]


class FAISSStateStorage:
    """FAISS-based storage and retrieval system for environment states."""
    
    def __init__(self, 
                 state_dim: int,
                 index_type: str = "IVF",
                 nlist: int = 100,
                 nprobe: int = 10,
                 use_gpu: bool = True):
        """
        Initialize FAISS state storage.
        
        Args:
            state_dim: Dimension of state vectors
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search for IVF index
            use_gpu: Whether to use GPU for FAISS operations
        """
        self.state_dim = state_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize FAISS index
        self.index = self._create_index()
        self.state_info: List[StateInfo] = []
        self.episode_states: Dict[int, List[int]] = defaultdict(list)
        
        # Performance metrics
        self.search_times: List[float] = []
        self.add_times: List[float] = []
        
    def _create_index(self) -> faiss.Index:
        """Create and configure FAISS index."""
        if self.index_type == "Flat":
            index = faiss.IndexFlatL2(self.state_dim)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.state_dim)
            index = faiss.IndexIVFFlat(quantizer, self.state_dim, self.nlist)
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.state_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print(f"Using GPU for FAISS index: {self.index_type}")
            except Exception as e:
                print(f"Failed to use GPU, falling back to CPU: {e}")
                self.use_gpu = False
        
        return index
    
    def add_state(self, 
                  state: np.ndarray, 
                  episode_id: int, 
                  step_id: int, 
                  reward: float = 0.0,
                  done: bool = False,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a state to the storage.
        
        Args:
            state: State vector to store
            episode_id: Episode identifier
            step_id: Step identifier within episode
            reward: Reward received at this state
            done: Whether episode ended at this state
            metadata: Additional metadata
            
        Returns:
            Index of the stored state
        """
        start_time = time.time()
        
        # Ensure state is in correct format
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        state = state.astype(np.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            # Collect enough states before training (need at least nlist points for IVF)
            min_training_points = max(self.nlist, 100)
            if len(self.state_info) >= min_training_points - 1:
                # Now we have enough data to train
                existing_states = np.array([info.state_vector for info in self.state_info])
                combined_states = np.vstack([existing_states, state])
                print(f"Training IVF index with {len(combined_states)} states...")
                self.index.train(combined_states)
            elif len(self.state_info) == 0:
                # First state, store without training
                print(f"Collecting states for IVF training... (need {min_training_points}, have {len(self.state_info) + 1})")
            else:
                # Still collecting training data
                print(f"Collecting states for IVF training... (need {min_training_points}, have {len(self.state_info) + 1})")
                return len(self.state_info)  # Return early, don't add to index yet
        
        # Add to index only if trained (or not IVF)
        state_id = len(self.state_info)
        if self.index_type != "IVF" or self.index.is_trained:
            self.index.add(state)
        else:
            # For untrained IVF, we'll add all states at once after training
            pass
        
        # Store state info
        state_info = StateInfo(
            state_vector=state.flatten(),
            episode_id=episode_id,
            step_id=step_id,
            reward=reward,
            done=done,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.state_info.append(state_info)
        self.episode_states[episode_id].append(state_id)
        
        add_time = time.time() - start_time
        self.add_times.append(add_time)
        
        return state_id
    
    def search_nearest(self, 
                      query_state: np.ndarray, 
                      k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[StateInfo]]:
        """
        Search for k nearest neighbors to query state.
        
        Args:
            query_state: Query state vector
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices, state_infos)
        """
        start_time = time.time()
        
        # Ensure query state is in correct format
        if isinstance(query_state, torch.Tensor):
            query_state = query_state.cpu().numpy()
        
        query_state = query_state.astype(np.float32)
        if query_state.ndim == 1:
            query_state = query_state.reshape(1, -1)
        
        # Set search parameters for IVF
        if self.index_type == "IVF":
            self.index.nprobe = self.nprobe
        
        # Search
        distances, indices = self.index.search(query_state, k)
        
        # Get state info for results
        state_infos = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.state_info):
                state_infos.append(self.state_info[idx])
        
        search_time = time.time() - start_time
        self.search_times.append(search_time)
        
        return distances[0], indices[0], state_infos
    
    def get_episode_states(self, episode_id: int) -> List[StateInfo]:
        """Get all states from a specific episode."""
        state_ids = self.episode_states[episode_id]
        return [self.state_info[idx] for idx in state_ids]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for FAISS operations."""
        stats = {
            'total_states': len(self.state_info),
            'total_episodes': len(self.episode_states),
            'index_type': self.index_type,
            'use_gpu': self.use_gpu,
        }
        
        if self.search_times:
            stats.update({
                'avg_search_time': np.mean(self.search_times),
                'min_search_time': np.min(self.search_times),
                'max_search_time': np.max(self.search_times),
                'total_searches': len(self.search_times),
            })
        
        if self.add_times:
            stats.update({
                'avg_add_time': np.mean(self.add_times),
                'min_add_time': np.min(self.add_times),
                'max_add_time': np.max(self.add_times),
                'total_adds': len(self.add_times),
            })
        
        return stats
    
    def save(self, filepath: str):
        """Save the state storage to disk."""
        # Save FAISS index
        if self.use_gpu:
            # Move to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, f"{filepath}.index")
        else:
            faiss.write_index(self.index, f"{filepath}.index")
        
        # Save state info and metadata
        save_data = {
            'state_info': self.state_info,
            'episode_states': dict(self.episode_states),
            'config': {
                'state_dim': self.state_dim,
                'index_type': self.index_type,
                'nlist': self.nlist,
                'nprobe': self.nprobe,
                'use_gpu': self.use_gpu,
            },
            'performance': {
                'search_times': self.search_times,
                'add_times': self.add_times,
            }
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """Load state storage from disk."""
        # Load FAISS index
        index = faiss.read_index(f"{filepath}.index")
        
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                print(f"Failed to load index on GPU: {e}")
                self.use_gpu = False
        
        self.index = index
        
        # Load state info and metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            save_data = pickle.load(f)
        
        self.state_info = save_data['state_info']
        self.episode_states = defaultdict(list, save_data['episode_states'])
        self.search_times = save_data['performance']['search_times']
        self.add_times = save_data['performance']['add_times']


def benchmark_faiss_performance(storage: FAISSStateStorage, 
                                query_states: np.ndarray, 
                                k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
    """
    Benchmark FAISS search performance.
    
    Args:
        storage: FAISS storage instance
        query_states: Array of query states for benchmarking
        k_values: List of k values to test
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for k in k_values:
        search_times = []
        
        for query_state in query_states:
            start_time = time.time()
            distances, indices, _ = storage.search_nearest(query_state, k)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        results[f'k_{k}'] = {
            'avg_time': np.mean(search_times),
            'min_time': np.min(search_times),
            'max_time': np.max(search_times),
            'std_time': np.std(search_times),
            'total_queries': len(search_times),
        }
    
    return results


def collect_and_store_states(env_name: str = "T1JoystickFlatTerrain", 
                           num_episodes: int = 10,
                           max_steps_per_episode: int = 1000,
                           save_path: str = "faiss_states",
                           index_type: str = "Flat") -> FAISSStateStorage:
    """
    Collect states from environment episodes and store in FAISS.
    
    Args:
        env_name: Name of the environment
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        save_path: Path to save the storage
        index_type: Type of FAISS index to use
        
    Returns:
        Populated FAISS storage instance
    """
    print(f"Collecting states from {env_name} for {num_episodes} episodes...")
    
    # Create environment
    envs, _, _ = make_env(env_name, seed=42, num_envs=1, num_eval_envs=1, device_rank=0)
    
    # Get observation dimension
    obs = envs.reset()
    if isinstance(obs, torch.Tensor):
        obs_dim = obs.shape[-1]
    else:
        obs_dim = len(obs)
    
    print(f"Observation dimension: {obs_dim}")
    
    # Initialize storage
    storage = FAISSStateStorage(
        state_dim=obs_dim,
        index_type=index_type,
        use_gpu=torch.cuda.is_available()
    )
    
    # Collect states
    total_states = 0
    for episode in range(num_episodes):
        obs = envs.reset()
        episode_reward = 0.0
        
        for step in range(max_steps_per_episode):
            # Convert observation to numpy if needed
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy().flatten()
            else:
                obs_np = np.array(obs).flatten()
            
            # Take random action
            if hasattr(envs, 'action_space'):
                action = envs.action_space.sample()
            else:
                # For environments without action_space, use random actions based on num_actions
                # Need to match the batch dimension of the environment (usually 1 for single env)
                action_dim = envs.num_actions
                action = torch.randn(1, action_dim)  # Shape: [batch_size, action_dim]
                
                # Ensure action is on the same device as observations
                if isinstance(obs, torch.Tensor):
                    action = action.to(obs.device)
            next_obs, reward, done, _ = envs.step(action)
            
            # Convert reward and done to scalars if they are tensors
            if isinstance(reward, torch.Tensor):
                reward_scalar = reward.item() if reward.numel() == 1 else reward.mean().item()
            else:
                reward_scalar = float(reward)
            
            if isinstance(done, torch.Tensor):
                done_scalar = done.item() if done.numel() == 1 else done.any().item()
            else:
                done_scalar = bool(done)
            
            # Store state
            storage.add_state(
                state=obs_np,
                episode_id=episode,
                step_id=step,
                reward=reward_scalar,
                done=done_scalar,
                metadata={'episode_reward': episode_reward}
            )
            
            obs = next_obs
            episode_reward += reward_scalar
            total_states += 1
            
            if done_scalar:
                break
        
        print(f"Episode {episode + 1}/{num_episodes} completed: {step + 1} steps, reward: {episode_reward:.2f}")
    
    print(f"Total states collected: {total_states}")
    
    # Save storage
    storage.save(save_path)
    print(f"Storage saved to {save_path}")
    
    return storage


class StateRestorationManager:
    """Manager for restoring environment states using FAISS nearest neighbor search."""
    
    def __init__(self, storage: FAISSStateStorage, env):
        """
        Initialize state restoration manager.
        
        Args:
            storage: FAISS storage instance
            env: Environment instance
        """
        self.storage = storage
        self.env = env
        self.restoration_times = []
        self.restoration_accuracy = []
        
    def restore_nearest_state(self, target_state: np.ndarray, k: int = 1) -> Tuple[Any, float, Dict]:
        """
        Restore environment to the nearest stored state.
        
        Args:
            target_state: Target state to find nearest neighbor for
            k: Number of nearest neighbors to consider
            
        Returns:
            Tuple of (restored_obs, restoration_time, restoration_info)
        """
        start_time = time.time()
        
        # Find nearest neighbor
        distances, indices, state_infos = self.storage.search_nearest(target_state, k)
        
        if len(state_infos) == 0:
            raise ValueError("No states found in storage")
        
        # Use the closest state
        closest_state_info = state_infos[0]
        closest_distance = distances[0]
        
        # Reset environment and attempt to restore state
        obs = self.env.reset()
        
        # For demonstration, we'll use the stored state as the "restored" observation
        # In practice, you would need environment-specific state restoration logic
        restored_obs = closest_state_info.state_vector
        
        restoration_time = time.time() - start_time
        self.restoration_times.append(restoration_time)
        
        restoration_info = {
            'target_state_norm': np.linalg.norm(target_state),
            'restored_state_norm': np.linalg.norm(restored_obs),
            'distance_to_target': closest_distance,
            'episode_id': closest_state_info.episode_id,
            'step_id': closest_state_info.step_id,
            'reward': closest_state_info.reward,
            'restoration_time': restoration_time,
            'k_considered': k
        }
        
        return restored_obs, restoration_time, restoration_info
    
    def benchmark_restoration(self, query_states: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """
        Benchmark state restoration performance.
        
        Args:
            query_states: Array of query states for benchmarking
            k_values: List of k values to test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for k in k_values:
            restoration_times = []
            distances = []
            
            for query_state in query_states:
                try:
                    restored_obs, rest_time, rest_info = self.restore_nearest_state(query_state, k)
                    restoration_times.append(rest_time)
                    distances.append(rest_info['distance_to_target'])
                except Exception as e:
                    print(f"Failed to restore state: {e}")
                    continue
            
            if restoration_times:
                results[f'k_{k}'] = {
                    'avg_restoration_time': np.mean(restoration_times),
                    'min_restoration_time': np.min(restoration_times),
                    'max_restoration_time': np.max(restoration_times),
                    'std_restoration_time': np.std(restoration_times),
                    'avg_distance': np.mean(distances),
                    'min_distance': np.min(distances),
                    'max_distance': np.max(distances),
                    'total_restorations': len(restoration_times),
                }
        
        return results
    
    def get_restoration_stats(self) -> Dict[str, float]:
        """Get restoration performance statistics."""
        stats = {}
        
        if self.restoration_times:
            stats.update({
                'avg_restoration_time': np.mean(self.restoration_times),
                'min_restoration_time': np.min(self.restoration_times),
                'max_restoration_time': np.max(self.restoration_times),
                'total_restorations': len(self.restoration_times),
            })
        
        return stats


def main():
    """Main function to demonstrate FAISS state storage and benchmarking."""
    args = get_args()
    
    # Collect and store states
    storage = collect_and_store_states(
        env_name=args.env_name,
        num_episodes=20,
        max_steps_per_episode=1000,
        save_path=f"faiss_states_{args.env_name}",
        index_type="Flat"
    )
    
    # Create environment for restoration testing
    envs, _, _ = make_env(args.env_name, seed=42, num_envs=1, num_eval_envs=1, device_rank=0)
    
    # Initialize state restoration manager
    restoration_manager = StateRestorationManager(storage, envs)
    
    # Generate random query states for benchmarking
    print("\nGenerating random query states for benchmarking...")
    query_states = np.random.randn(100, storage.state_dim).astype(np.float32)
    
    # Benchmark FAISS search performance
    print("\nBenchmarking FAISS search performance...")
    search_benchmark_results = benchmark_faiss_performance(
        storage=storage,
        query_states=query_states,
        k_values=[1, 5, 10, 20, 50]
    )
    
    # Benchmark state restoration performance
    print("\nBenchmarking state restoration performance...")
    restoration_benchmark_results = restoration_manager.benchmark_restoration(
        query_states=query_states[:20],  # Use fewer states for restoration benchmark
        k_values=[1, 5, 10]
    )
    
    # Print search results
    print("\n" + "="*60)
    print("FAISS SEARCH PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    
    for k, results in search_benchmark_results.items():
        print(f"\nK = {k.split('_')[1]}:")
        print(f"  Average search time: {results['avg_time']*1000:.3f} ms")
        print(f"  Min search time: {results['min_time']*1000:.3f} ms")
        print(f"  Max search time: {results['max_time']*1000:.3f} ms")
        print(f"  Std search time: {results['std_time']*1000:.3f} ms")
    
    # Print restoration results
    print("\n" + "="*60)
    print("STATE RESTORATION BENCHMARK RESULTS")
    print("="*60)
    
    for k, results in restoration_benchmark_results.items():
        print(f"\nK = {k.split('_')[1]}:")
        print(f"  Average restoration time: {results['avg_restoration_time']*1000:.3f} ms")
        print(f"  Min restoration time: {results['min_restoration_time']*1000:.3f} ms")
        print(f"  Max restoration time: {results['max_restoration_time']*1000:.3f} ms")
        print(f"  Average distance to target: {results['avg_distance']:.4f}")
        print(f"  Min distance to target: {results['min_distance']:.4f}")
        print(f"  Max distance to target: {results['max_distance']:.4f}")
    
    # Overall statistics
    stats = storage.get_performance_stats()
    print(f"\n" + "="*60)
    print("OVERALL STORAGE STATISTICS")
    print("="*60)
    print(f"Total states stored: {stats['total_states']}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Index type: {stats['index_type']}")
    print(f"Using GPU: {stats['use_gpu']}")
    
    if 'avg_search_time' in stats:
        print(f"Average search time: {stats['avg_search_time']*1000:.3f} ms")
    if 'avg_add_time' in stats:
        print(f"Average add time: {stats['avg_add_time']*1000:.3f} ms")
    
    # Demonstrate nearest neighbor search
    print(f"\n" + "="*60)
    print("NEAREST NEIGHBOR SEARCH DEMONSTRATION")
    print("="*60)
    
    # Use first stored state as query
    if storage.state_info:
        query_state = storage.state_info[0].state_vector
        distances, indices, state_infos = storage.search_nearest(query_state, k=5)
        
        print(f"Query state from episode {storage.state_info[0].episode_id}, step {storage.state_info[0].step_id}")
        print("Nearest neighbors:")
        for i, (dist, idx, info) in enumerate(zip(distances, indices, state_infos)):
            print(f"  {i+1}. Distance: {dist:.4f}, Episode: {info.episode_id}, Step: {info.step_id}, Reward: {info.reward:.3f}")
    
    # Demonstrate state restoration
    print(f"\n" + "="*60)
    print("STATE RESTORATION DEMONSTRATION")
    print("="*60)
    
    if len(storage.state_info) > 10:
        # Use a state from the middle of storage as target
        target_state = storage.state_info[len(storage.state_info)//2].state_vector
        
        try:
            restored_obs, rest_time, rest_info = restoration_manager.restore_nearest_state(target_state, k=3)
            print(f"Target state from episode {storage.state_info[len(storage.state_info)//2].episode_id}")
            print(f"Restoration time: {rest_time*1000:.3f} ms")
            print(f"Distance to target: {rest_info['distance_to_target']:.4f}")
            print(f"Restored from episode {rest_info['episode_id']}, step {rest_info['step_id']}")
            print(f"Restored state reward: {rest_info['reward']:.3f}")
        except Exception as e:
            print(f"Failed to demonstrate restoration: {e}")
    
    print(f"\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()