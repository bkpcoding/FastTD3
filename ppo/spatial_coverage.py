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