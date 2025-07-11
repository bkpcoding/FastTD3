"""Spatial coverage metric using UMAP projection and grid-based coverage calculation."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import umap
from collections import deque


class SpatialCoverageMetric:
    """
    Computes spatial coverage of batch data using UMAP projection and grid-based analysis.
    
    This metric quantifies the cumulative dispersion of batch data across iterations by:
    1. Projecting high-dimensional data to 2D using UMAP
    2. Partitioning the 2D space into a uniform grid
    3. Computing coverage as the fraction of grid cells visited
    """
    
    def __init__(
        self,
        grid_size: int = 32,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        buffer_size: int = 10000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the spatial coverage metric.
        
        Args:
            grid_size: Size of the grid (G x G) for coverage calculation
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random seed for reproducibility
            buffer_size: Maximum number of data points to keep in buffer
            device: PyTorch device for tensor operations
        """
        self.grid_size = grid_size
        self.device = device or torch.device('cpu')
        self.buffer_size = buffer_size
        
        # UMAP parameters
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        
        # Data buffer to accumulate batch data across iterations
        self.data_buffer = deque(maxlen=buffer_size)
        
        # UMAP reducer (will be fitted when we have enough data)
        self.umap_reducer = None
        self.is_fitted = False
        
        # Coverage tracking
        self.occupied_cells = set()
        self.coverage_history = []
        
    def add_batch_data(self, batch_data: torch.Tensor) -> None:
        """
        Add a batch of data to the buffer.
        
        Args:
            batch_data: Tensor of shape (batch_size, feature_dim)
        """
        if batch_data.device != self.device:
            batch_data = batch_data.to(self.device)
            
        # Convert to numpy for UMAP processing
        batch_numpy = batch_data.detach().cpu().numpy()
        
        # Add each sample to the buffer
        for sample in batch_numpy:
            self.data_buffer.append(sample)
    
    def _fit_umap(self) -> None:
        """Fit UMAP on the accumulated data."""
        if len(self.data_buffer) < max(50, self.n_neighbors + 1):
            return  # Not enough data to fit UMAP
            
        # Convert buffer to numpy array
        data_array = np.array(list(self.data_buffer))
        
        # Initialize and fit UMAP
        self.umap_reducer = umap.UMAP(
            n_neighbors=min(self.n_neighbors, len(data_array) - 1),
            min_dist=self.min_dist,
            n_components=2,
            random_state=self.random_state,
            verbose=False
        )
        
        try:
            self.umap_reducer.fit(data_array)
            self.is_fitted = True
        except Exception as e:
            print(f"Warning: UMAP fitting failed: {e}")
            self.is_fitted = False
    
    def _project_to_2d(self, data: np.ndarray) -> np.ndarray:
        """Project data to 2D using fitted UMAP."""
        if not self.is_fitted:
            self._fit_umap()
            
        if not self.is_fitted:
            # Fallback: use first two dimensions if UMAP fails
            return data[:, :2] if data.shape[1] >= 2 else np.zeros((data.shape[0], 2))
            
        try:
            projection = self.umap_reducer.transform(data)
            # Normalize to [0, 1) range
            projection_min = projection.min(axis=0)
            projection_max = projection.max(axis=0)
            projection_range = projection_max - projection_min
            
            # Avoid division by zero
            projection_range = np.where(projection_range == 0, 1, projection_range)
            
            normalized = (projection - projection_min) / projection_range
            # Ensure values are in [0, 1) range
            normalized = np.clip(normalized, 0, 0.999)
            
            return normalized
        except Exception as e:
            print(f"Warning: UMAP transform failed: {e}")
            # Fallback: use first two dimensions
            return data[:, :2] if data.shape[1] >= 2 else np.zeros((data.shape[0], 2))
    
    def _compute_grid_indices(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Compute grid cell indices for 2D points.
        
        Args:
            points_2d: Array of shape (N, 2) with values in [0, 1)
            
        Returns:
            Array of shape (N, 2) with grid cell indices
        """
        # Compute grid indices: idx = min(floor(coord * G), G-1)
        indices = np.floor(points_2d * self.grid_size).astype(int)
        indices = np.clip(indices, 0, self.grid_size - 1)
        return indices
    
    def update_coverage(self) -> float:
        """
        Update coverage metric with current buffer data.
        
        Returns:
            Current coverage value (fraction of grid cells occupied)
        """
        if len(self.data_buffer) == 0:
            return 0.0
            
        # Convert buffer to array
        data_array = np.array(list(self.data_buffer))
        
        # Project to 2D
        points_2d = self._project_to_2d(data_array)
        
        # Compute grid indices
        grid_indices = self._compute_grid_indices(points_2d)
        
        # Update occupied cells set
        for idx in grid_indices:
            self.occupied_cells.add(tuple(idx))
        
        # Compute coverage
        total_cells = self.grid_size ** 2
        coverage = len(self.occupied_cells) / total_cells
        
        # Store in history
        self.coverage_history.append(coverage)
        
        return coverage
    
    def get_current_coverage(self) -> float:
        """Get the current coverage value."""
        if not self.coverage_history:
            return self.update_coverage()
        return self.coverage_history[-1]
    
    def get_coverage_history(self) -> List[float]:
        """Get the history of coverage values."""
        return self.coverage_history.copy()
    
    def reset(self) -> None:
        """Reset the coverage metric."""
        self.data_buffer.clear()
        self.occupied_cells.clear()
        self.coverage_history.clear()
        self.umap_reducer = None
        self.is_fitted = False
    
    def visualize_coverage(
        self, 
        title: str = "Spatial Coverage Visualization",
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Create visualization of the spatial coverage.
        
        Args:
            title: Title for the plot
            figsize: Figure size
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        if len(self.data_buffer) == 0:
            print("No data available for visualization")
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: 2D projection with grid overlay
        data_array = np.array(list(self.data_buffer))
        points_2d = self._project_to_2d(data_array)
        
        # Plot data points
        ax1.scatter(points_2d[:, 0], points_2d[:, 1], alpha=0.6, s=1)
        
        # Draw grid
        for i in range(self.grid_size + 1):
            coord = i / self.grid_size
            ax1.axhline(y=coord, color='gray', alpha=0.3, linewidth=0.5)
            ax1.axvline(x=coord, color='gray', alpha=0.3, linewidth=0.5)
        
        # Highlight occupied cells
        for cell in self.occupied_cells:
            x, y = cell
            rect_x = x / self.grid_size
            rect_y = y / self.grid_size
            rect_width = 1 / self.grid_size
            rect_height = 1 / self.grid_size
            
            rect = plt.Rectangle(
                (rect_x, rect_y), rect_width, rect_height,
                fill=True, alpha=0.3, color='red'
            )
            ax1.add_patch(rect)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('UMAP Dimension 1')
        ax1.set_ylabel('UMAP Dimension 2')
        ax1.set_title('2D Projection with Grid Coverage')
        ax1.set_aspect('equal')
        
        # Right plot: Coverage history
        if len(self.coverage_history) > 1:
            ax2.plot(self.coverage_history, linewidth=2)
            ax2.set_xlabel('Update Step')
            ax2.set_ylabel('Coverage')
            ax2.set_title('Coverage Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'Not enough data\nfor history plot', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Coverage History')
        
        plt.suptitle(f"{title}\nCurrent Coverage: {self.get_current_coverage():.3f}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_summary(self) -> dict:
        """Get a summary of the coverage metrics."""
        current_coverage = self.get_current_coverage()
        
        summary = {
            'current_coverage': current_coverage,
            'total_cells': self.grid_size ** 2,
            'occupied_cells': len(self.occupied_cells),
            'data_points': len(self.data_buffer),
            'grid_size': self.grid_size,
            'is_umap_fitted': self.is_fitted,
        }
        
        if self.coverage_history:
            summary.update({
                'max_coverage': max(self.coverage_history),
                'coverage_growth': current_coverage - self.coverage_history[0] if len(self.coverage_history) > 1 else 0.0,
                'update_steps': len(self.coverage_history)
            })
            
        return summary
    
    def save_final_coverage(self, save_path: str, num_envs: int) -> None:
        """
        Save the final coverage state with number of environments in filename.
        
        Args:
            save_path: Base path for saving (without extension)
            num_envs: Number of environments used in training
        """
        import pickle
        import os
        
        # Create filename with num_envs
        base_name = os.path.basename(save_path)
        dir_name = os.path.dirname(save_path)
        filename = f"{base_name}_coverage_numenvs_{num_envs}.pkl"
        full_path = os.path.join(dir_name, filename)
        
        # Prepare data to save
        coverage_data = {
            'final_coverage': self.get_current_coverage(),
            'coverage_history': self.get_coverage_history(),
            'summary': self.get_summary(),
            'num_envs': num_envs,
            'occupied_cells': list(self.occupied_cells),
            'grid_size': self.grid_size,
            'total_data_points': len(self.data_buffer)
        }
        
        # Save to pickle file
        os.makedirs(dir_name, exist_ok=True)
        with open(full_path, 'wb') as f:
            pickle.dump(coverage_data, f)
        
        print(f"Final coverage data saved to: {full_path}")
        print(f"Final coverage: {coverage_data['final_coverage']:.4f}")
        print(f"Occupied cells: {coverage_data['summary']['occupied_cells']}/{coverage_data['summary']['total_cells']}")
        
        return full_path


def visualize_faiss_states_by_agent(
    faiss_storage,
    experiment_dir: str,
    grid_size: int = 32,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8)
) -> List[str]:
    """
    Visualize FAISS states with different colors for different agents in separate plots.
    
    Args:
        faiss_storage: FAISS storage object containing states and metadata
        experiment_dir: Directory to save the visualization
        grid_size: Size of the grid for coverage calculation
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed for reproducibility
        figsize: Figure size for each plot
        
    Returns:
        List of paths to saved visualizations
    """
    import os
    
    if len(faiss_storage.state_info) == 0:
        print("No FAISS states available for visualization")
        return []
    
    # Extract states and agent information
    states = []
    agent_ids = []
    rewards = []
    
    for state_info in faiss_storage.state_info:
        states.append(state_info.state_vector)
        # Extract agent_id from metadata, default to "unknown" if not found
        agent_id = state_info.metadata.get("agent_id", "unknown") if state_info.metadata else "unknown"
        agent_ids.append(agent_id)
        rewards.append(state_info.reward)
    
    states_array = np.array(states)
    agent_ids = np.array(agent_ids)
    rewards = np.array(rewards)
    
    print(f"Visualizing {len(states)} FAISS states from agents: {np.unique(agent_ids)}")
    
    # Create UMAP projection
    umap_reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(states) - 1),
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=False
    )
    
    try:
        points_2d = umap_reducer.fit_transform(states_array)
        
        # Normalize to [0, 1] range
        points_min = points_2d.min(axis=0)
        points_max = points_2d.max(axis=0)
        points_range = points_max - points_min
        points_range = np.where(points_range == 0, 1, points_range)
        points_2d_norm = (points_2d - points_min) / points_range
        
    except Exception as e:
        print(f"UMAP failed: {e}, using first two dimensions")
        points_2d_norm = states_array[:, :2] if states_array.shape[1] >= 2 else np.random.random((len(states), 2))
        # Normalize to [0, 1] range
        points_min = points_2d_norm.min(axis=0)
        points_max = points_2d_norm.max(axis=0)
        points_range = points_max - points_min
        points_range = np.where(points_range == 0, 1, points_range)
        points_2d_norm = (points_2d_norm - points_min) / points_range
    
    # Define colors for different agents
    unique_agents = np.unique(agent_ids)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_agents)))
    agent_color_map = {agent: color for agent, color in zip(unique_agents, colors)}
    
    saved_plots = []
    
    # Plot 1: States by agent
    plt.figure(figsize=figsize)
    for agent in unique_agents:
        mask = agent_ids == agent
        agent_points = points_2d_norm[mask]
        
        plt.scatter(
            agent_points[:, 0], 
            agent_points[:, 1], 
            c=[agent_color_map[agent]], 
            label=f'Agent {agent} ({np.sum(mask)} states)',
            alpha=0.7,
            s=30
        )
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('FAISS States by Agent', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    agent_plot_path = os.path.join(experiment_dir, "faiss_states_by_agent.png")
    plt.savefig(agent_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(agent_plot_path)
    
    # Plot 2: States colored by reward
    plt.figure(figsize=figsize)
    
    scatter = plt.scatter(
        points_2d_norm[:, 0], 
        points_2d_norm[:, 1], 
        c=rewards, 
        cmap='viridis',
        alpha=0.7,
        s=30
    )
    plt.colorbar(scatter, label='Reward')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('FAISS States by Reward', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    reward_plot_path = os.path.join(experiment_dir, "faiss_states_by_reward.png")
    plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(reward_plot_path)
    
    # Plot 3: Grid coverage by agent
    plt.figure(figsize=figsize)
    
    # Create combined grid for visualization
    combined_grid = np.zeros((grid_size, grid_size, 3))  # RGB
    
    for i, agent in enumerate(unique_agents):
        mask = agent_ids == agent
        agent_points = points_2d_norm[mask]
        
        if len(agent_points) > 0:
            # Compute grid indices
            grid_indices = np.floor(agent_points * grid_size).astype(int)
            grid_indices = np.clip(grid_indices, 0, grid_size - 1)
            
            # Use different colors for each agent
            agent_color = colors[i][:3]  # RGB values
            
            for idx in grid_indices:
                combined_grid[idx[1], idx[0]] = agent_color
    
    plt.imshow(combined_grid, extent=[0, 1, 0, 1], origin='lower', alpha=0.8)
    
    # Draw grid lines
    for i in range(grid_size + 1):
        coord = i / grid_size
        plt.axhline(y=coord, color='gray', alpha=0.3, linewidth=0.5)
        plt.axvline(x=coord, color='gray', alpha=0.3, linewidth=0.5)
    
    # Add legend for agents
    legend_elements = []
    for agent in unique_agents:
        mask = agent_ids == agent
        color = agent_color_map[agent]
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'Agent {agent}'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Grid Coverage by Agent', fontsize=14, fontweight='bold')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    coverage_plot_path = os.path.join(experiment_dir, "faiss_grid_coverage.png")
    plt.savefig(coverage_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(coverage_plot_path)
    
    # Plot 4: Statistics summary
    plt.figure(figsize=(10, 12))
    plt.axis('off')
    
    # Compute statistics
    total_states = len(states)
    total_cells = grid_size ** 2
    
    # Overall coverage
    all_grid_indices = np.floor(points_2d_norm * grid_size).astype(int)
    all_grid_indices = np.clip(all_grid_indices, 0, grid_size - 1)
    occupied_cells = set()
    for idx in all_grid_indices:
        occupied_cells.add(tuple(idx))
    overall_coverage = len(occupied_cells) / total_cells
    
    # Per-agent statistics
    stats_text = f"FAISS States Spatial Coverage Analysis\n"
    stats_text += f"{'='*50}\n\n"
    stats_text += f"OVERALL STATISTICS:\n"
    stats_text += f"  Total States: {total_states}\n"
    stats_text += f"  Grid Size: {grid_size}x{grid_size} ({total_cells} cells)\n"
    stats_text += f"  Overall Coverage: {overall_coverage:.3f}\n"
    stats_text += f"  Occupied Cells: {len(occupied_cells)}\n"
    stats_text += f"  Unique Agents: {len(unique_agents)}\n\n"
    
    stats_text += f"PER-AGENT BREAKDOWN:\n"
    stats_text += f"{'-'*30}\n"
    
    for agent in unique_agents:
        mask = agent_ids == agent
        agent_count = np.sum(mask)
        agent_rewards = rewards[mask]
        agent_rewards_mean = np.mean(agent_rewards)
        agent_rewards_std = np.std(agent_rewards)
        agent_rewards_min = np.min(agent_rewards)
        agent_rewards_max = np.max(agent_rewards)
        
        # Agent-specific coverage
        agent_points = points_2d_norm[mask]
        if len(agent_points) > 0:
            agent_grid_indices = np.floor(agent_points * grid_size).astype(int)
            agent_grid_indices = np.clip(agent_grid_indices, 0, grid_size - 1)
            agent_occupied = set()
            for idx in agent_grid_indices:
                agent_occupied.add(tuple(idx))
            agent_coverage = len(agent_occupied) / total_cells
            agent_coverage_pct = (len(agent_occupied) / len(occupied_cells)) * 100
        else:
            agent_coverage = 0.0
            agent_coverage_pct = 0.0
        
        stats_text += f"\nAgent {agent}:\n"
        stats_text += f"  States Collected: {agent_count}\n"
        stats_text += f"  Spatial Coverage: {agent_coverage:.3f}\n"
        stats_text += f"  Unique Cells: {len(agent_occupied) if len(agent_points) > 0 else 0}\n"
        stats_text += f"  Coverage Contribution: {agent_coverage_pct:.1f}%\n"
        stats_text += f"  Reward Statistics:\n"
        stats_text += f"    Mean: {agent_rewards_mean:.2f}\n"
        stats_text += f"    Std:  {agent_rewards_std:.2f}\n"
        stats_text += f"    Min:  {agent_rewards_min:.2f}\n"
        stats_text += f"    Max:  {agent_rewards_max:.2f}\n"
    
    # Add UMAP parameters
    stats_text += f"\nUMAP PARAMETERS:\n"
    stats_text += f"{'-'*20}\n"
    stats_text += f"  n_neighbors: {n_neighbors}\n"
    stats_text += f"  min_dist: {min_dist}\n"
    stats_text += f"  random_state: {random_state}\n"
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=11)
    
    plt.title('FAISS States Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    stats_plot_path = os.path.join(experiment_dir, "faiss_analysis_summary.png")
    plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(stats_plot_path)
    
    print(f"FAISS spatial coverage visualizations saved:")
    for plot_path in saved_plots:
        print(f"  - {os.path.basename(plot_path)}")
    print(f"Total states visualized: {total_states}")
    print(f"Overall spatial coverage: {overall_coverage:.3f}")
    
    return saved_plots