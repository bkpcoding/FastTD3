#!/usr/bin/env python3
"""
Script to visualize replay buffer observation evolution over training time using UMAP embeddings.
Creates a GIF showing how the distribution of observations changes during training.
"""

import os
import glob
import pickle
import re
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import umap
from tqdm import tqdm


def load_buffer_snapshots(output_dir, run_name=None):
    """Load all buffer snapshots from the output directory."""
    
    # Find all buffer snapshot files
    if run_name:
        pattern = f"{output_dir}/{run_name}_buffer_snapshot_*.pkl"
    else:
        pattern = f"{output_dir}/*_buffer_snapshot_*.pkl"
    
    snapshot_files = glob.glob(pattern)
    snapshot_files.sort()
    
    if not snapshot_files:
        raise ValueError(f"No buffer snapshot files found in {output_dir}")
    
    snapshots = []
    for file_path in snapshot_files:
        # Extract global step from filename
        match = re.search(r'_buffer_snapshot_(\d+)\.pkl$', file_path)
        if match:
            global_step = int(match.group(1))
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Determine if this is a single agent run based on filename
            is_single_agent = '_hyperparams__1_' in file_path
            
            snapshots.append({
                'global_step': global_step,
                'observations': data['observations'],
                'rewards': data.get('rewards', None),  # Handle backward compatibility
                'metadata': data['_metadata'],
                'file_path': file_path,
                'is_single_agent': is_single_agent
            })
    
    # Sort by global step
    snapshots.sort(key=lambda x: x['global_step'])
    print(f"Loaded {len(snapshots)} buffer snapshots")
    
    return snapshots


def prepare_observations_for_umap(snapshots, max_samples_per_checkpoint=5000, top_percentile=10):
    """
    Prepare observations for UMAP by sampling from each checkpoint.
    Groups snapshots by global_step to handle multiple checkpoints at same step.
    Only uses top percentile of observations based on rewards.
    Returns combined observations and checkpoint labels.
    """
    # Group snapshots by global_step
    step_groups = {}
    for i, snapshot in enumerate(snapshots):
        global_step = snapshot['global_step']
        if global_step not in step_groups:
            step_groups[global_step] = []
        step_groups[global_step].append((i, snapshot))
    
    all_observations = []
    checkpoint_labels = []
    checkpoint_info = []
    
    checkpoint_idx = 0
    for global_step in sorted(step_groups.keys()):
        group_snapshots = step_groups[global_step]
        
        # Combine observations from all snapshots at this global step
        combined_obs_for_step = []
        combined_rewards_for_step = []
        combined_agent_types = []
        total_samples = 0
        
        for orig_idx, snapshot in group_snapshots:
            obs = snapshot['observations']  # Shape: (n_env, buffer_size, obs_dim)
            rewards = snapshot['rewards']  # Shape: (n_env, buffer_size)
            is_single_agent = snapshot['is_single_agent']
            
            obs_reshaped = obs.reshape(-1, obs.shape[-1])
            rewards_reshaped = rewards.reshape(-1) if rewards is not None else None
            
            # Create agent type array for each observation
            agent_type_array = np.full(obs_reshaped.shape[0], is_single_agent, dtype=bool)
            
            combined_obs_for_step.append(obs_reshaped)
            combined_agent_types.append(agent_type_array)
            if rewards_reshaped is not None:
                combined_rewards_for_step.append(rewards_reshaped)
            total_samples += obs_reshaped.shape[0]
        
        # Combine all observations, rewards, and agent types for this step
        step_observations = np.vstack(combined_obs_for_step)
        step_agent_types = np.concatenate(combined_agent_types)
        
        if combined_rewards_for_step:
            step_rewards = np.concatenate(combined_rewards_for_step)
            
            # Filter to top percentile based on rewards
            reward_threshold = np.percentile(step_rewards, 100 - top_percentile)
            top_reward_mask = step_rewards >= reward_threshold
            
            # Apply mask to get top reward observations and their agent types
            filtered_observations = step_observations[top_reward_mask]
            filtered_agent_types = step_agent_types[top_reward_mask]
            print(f"  Filtered to top {top_percentile}%: {filtered_observations.shape[0]} samples (threshold: {reward_threshold:.3f})")
        else:
            # Fallback if no rewards available - use all observations
            filtered_observations = step_observations
            filtered_agent_types = step_agent_types
            print(f"  No rewards available - using all {filtered_observations.shape[0]} samples")
        
        # Sample random subset to keep computation manageable
        n_samples = min(max_samples_per_checkpoint, filtered_observations.shape[0])
        if n_samples < filtered_observations.shape[0]:
            indices = np.random.choice(filtered_observations.shape[0], n_samples, replace=False)
            sampled_obs = filtered_observations[indices]
            sampled_agent_types = filtered_agent_types[indices]
        else:
            sampled_obs = filtered_observations
            sampled_agent_types = filtered_agent_types
        
        all_observations.append(sampled_obs)
        checkpoint_labels.extend([checkpoint_idx] * n_samples)
        checkpoint_info.append({
            'checkpoint_idx': checkpoint_idx,
            'global_step': global_step,
            'n_samples': n_samples,
            'n_original_snapshots': len(group_snapshots),
            'agent_types': sampled_agent_types
        })
        
        snapshot_info = f" (from {len(group_snapshots)} snapshots)" if len(group_snapshots) > 1 else ""
        print(f"Checkpoint {checkpoint_idx} (step {global_step}): {n_samples} samples{snapshot_info}")
        
        checkpoint_idx += 1
    
    # Combine all observations and agent types
    combined_observations = np.vstack(all_observations)
    checkpoint_labels = np.array(checkpoint_labels)
    
    # Create combined agent type array
    all_agent_types = []
    for info in checkpoint_info:
        all_agent_types.extend(info['agent_types'])
    agent_type_labels = np.array(all_agent_types)
    
    print(f"Total samples for UMAP: {combined_observations.shape[0]}")
    print(f"Unique global steps: {len(checkpoint_info)}")
    print(f"Single agent samples: {np.sum(agent_type_labels)}")
    print(f"Multi agent samples: {np.sum(~agent_type_labels)}")
    return combined_observations, checkpoint_labels, checkpoint_info, agent_type_labels


def compute_umap_embedding(observations, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """Compute UMAP embedding of observations."""
    print("Computing UMAP embedding...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        verbose=True
    )
    
    embedding = reducer.fit_transform(observations)
    print(f"UMAP embedding shape: {embedding.shape}")
    
    return embedding, reducer


def create_frame(embedding, checkpoint_labels, checkpoint_info, agent_type_labels, current_checkpoint_idx, 
                base_alpha=0.15, current_alpha=0.8, figsize=(10, 8)):
    """Create a single frame for the GIF."""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Define colors for agent types
    single_agent_color = '#FF6B6B'  # Red for single agent
    multi_agent_color = '#4ECDC4'   # Teal for multi agent
    
    # Plot all previous checkpoints with reduced opacity
    legend_added = {'single': False, 'multi': False}
    
    for i in range(current_checkpoint_idx + 1):
        checkpoint_mask = checkpoint_labels == i
        if not np.any(checkpoint_mask):
            continue
            
        alpha = current_alpha if i == current_checkpoint_idx else base_alpha
        
        # Get observations for this checkpoint
        checkpoint_embedding = embedding[checkpoint_mask]
        checkpoint_agent_types = agent_type_labels[checkpoint_mask]
        
        # Plot single agent points
        single_agent_mask = checkpoint_agent_types
        if np.any(single_agent_mask):
            label = "Single Agent" if (i == current_checkpoint_idx and not legend_added['single']) else None
            if label:
                legend_added['single'] = True
            ax.scatter(
                checkpoint_embedding[single_agent_mask, 0], 
                checkpoint_embedding[single_agent_mask, 1],
                c=single_agent_color, 
                alpha=alpha,
                s=10,
                label=label
            )
        
        # Plot multi agent points
        multi_agent_mask = ~checkpoint_agent_types
        if np.any(multi_agent_mask):
            label = "Multi Agent" if (i == current_checkpoint_idx and not legend_added['multi']) else None
            if label:
                legend_added['multi'] = True
            ax.scatter(
                checkpoint_embedding[multi_agent_mask, 0], 
                checkpoint_embedding[multi_agent_mask, 1],
                c=multi_agent_color, 
                alpha=alpha,
                s=10,
                label=label
            )
    
    # Customize plot
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(f'Replay Buffer Observations Evolution\nCurrent: Step {checkpoint_info[current_checkpoint_idx]["global_step"]}', 
                 fontsize=14, fontweight='bold')
    
    if current_checkpoint_idx >= 0:
        ax.legend(loc='upper right', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    # Set consistent axis limits
    ax.set_xlim(embedding[:, 0].min() - 1, embedding[:, 0].max() + 1)
    ax.set_ylim(embedding[:, 1].min() - 1, embedding[:, 1].max() + 1)
    
    plt.tight_layout()
    
    # Convert to image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image


def create_buffer_evolution_gif(output_dir, run_name=None, max_samples=5000, 
                               output_gif_path=None, frame_duration=800, top_percentile=10):
    """
    Main function to create GIF showing buffer observation evolution.
    """
    
    # Load snapshots
    print("Loading buffer snapshots...")
    snapshots = load_buffer_snapshots(output_dir, run_name)
    
    if len(snapshots) < 2:
        raise ValueError("Need at least 2 snapshots to create evolution GIF")
    
    # Prepare observations for UMAP
    print("Preparing observations...")
    observations, checkpoint_labels, checkpoint_info, agent_type_labels = prepare_observations_for_umap(
        snapshots, max_samples, top_percentile
    )
    
    # Compute UMAP embedding
    embedding, reducer = compute_umap_embedding(observations)
    
    # Create frames
    print("Creating GIF frames...")
    frames = []
    
    for i in tqdm(range(len(checkpoint_info)), desc="Generating frames"):
        frame = create_frame(embedding, checkpoint_labels, checkpoint_info, agent_type_labels, i)
        frames.append(Image.fromarray(frame))
    
    # Add a few extra frames at the end to pause on final result
    for _ in range(3):
        frames.append(frames[-1])
    
    # Save GIF
    if output_gif_path is None:
        if run_name:
            output_gif_path = f"{output_dir}/{run_name}_buffer_evolution.gif"
        else:
            output_gif_path = f"{output_dir}/buffer_evolution.gif"
    
    print(f"Saving GIF to {output_gif_path}")
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )
    
    # Save final frame as PNG
    final_png_path = output_gif_path.replace('.gif', '_final.png')
    frames[-4].save(final_png_path)  # Use -4 to get actual final frame before duplicates
    print(f"Final frame saved as PNG: {final_png_path}")
    
    print(f"GIF created successfully: {output_gif_path}")
    print(f"Total frames: {len(frames)}")
    
    return output_gif_path


def main():
    parser = argparse.ArgumentParser(description="Visualize replay buffer observation evolution")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory containing buffer snapshots")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Specific run name to analyze (optional)")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="Maximum samples per checkpoint for UMAP")
    parser.add_argument("--output_gif", type=str, default=None,
                       help="Output GIF path (auto-generated if not specified)")
    parser.add_argument("--frame_duration", type=int, default=2000,
                       help="Duration of each frame in milliseconds")
    parser.add_argument("--umap_neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter")
    parser.add_argument("--top_percentile", type=float, default=10.0,
                       help="Only plot top percentile of states based on rewards")
    
    args = parser.parse_args()
    
    # Create the GIF
    gif_path = create_buffer_evolution_gif(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_samples=args.max_samples,
        output_gif_path=args.output_gif,
        frame_duration=args.frame_duration,
        top_percentile=args.top_percentile
    )
    
    print(f"\nVisualization complete! GIF saved to: {gif_path}")


if __name__ == "__main__":
    main()