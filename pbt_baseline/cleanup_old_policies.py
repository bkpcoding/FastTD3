#!/usr/bin/env python3
"""Utility script to clean up PBT policies based on performance rankings."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import PBT modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbt_baseline.pbt import PbtObserver


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up PBT policies based on performance rankings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean up bottom 25% of policies based on performance
  python cleanup_old_policies.py --cleanup-bottom-performers

  # Keep only specific policies (remove all others)
  python cleanup_old_policies.py --keep-policies 0 1 2

  # Clean up policies that haven't been updated recently
  python cleanup_old_policies.py --cleanup-stale --max-stale-hours 48

  # Dry run to see what would be removed
  python cleanup_old_policies.py --cleanup-bottom-performers --dry-run
        """
    )
    
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="./train_dir/pbt_workspace",
        help="PBT workspace directory (default: ./train_dir/pbt_workspace)"
    )
    
    parser.add_argument(
        "--keep-policies",
        type=int,
        nargs="+",
        help="Policy indices to keep (remove all others)"
    )
    
    parser.add_argument(
        "--cleanup-bottom-performers",
        action="store_true",
        help="Clean up bottom performing policies based on PBT rankings"
    )
    
    parser.add_argument(
        "--cleanup-stale",
        action="store_true",
        help="Clean up policies that haven't been updated recently"
    )
    
    parser.add_argument(
        "--max-stale-hours",
        type=float,
        default=48.0,
        help="Consider policies stale if not updated for this many hours (default: 48)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


class MockArgs:
    """Mock args class for PbtObserver initialization."""
    def __init__(self, workspace_dir):
        self.pbt_workspace = os.path.basename(workspace_dir)
        self.pbt_num_policies = 10  # Default, will be overridden
        

def get_existing_policies(workspace_dir):
    """Get list of existing policy indices from workspace directory."""
    policies = []
    if os.path.exists(workspace_dir):
        for item in os.listdir(workspace_dir):
            if item.startswith('policy_') and os.path.isdir(os.path.join(workspace_dir, item)):
                try:
                    policy_idx = int(item.split('_')[1])
                    policies.append(policy_idx)
                except (ValueError, IndexError):
                    pass
    return sorted(policies)


def main():
    """Main cleanup function."""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Resolve workspace directory
    workspace_dir = os.path.abspath(args.workspace_dir)
    train_dir = os.path.dirname(workspace_dir)
    
    logger.info(f"PBT workspace directory: {workspace_dir}")
    logger.info(f"Training directory: {train_dir}")
    
    if not os.path.exists(workspace_dir):
        logger.warning(f"Workspace directory does not exist: {workspace_dir}")
        return
    
    # Get existing policies
    existing_policies = get_existing_policies(workspace_dir)
    logger.info(f"Found existing policies: {existing_policies}")
    
    if not existing_policies:
        logger.info("No policies found, nothing to clean up")
        return
    
    # Create mock args for PbtObserver
    mock_args = MockArgs(workspace_dir)
    mock_args.pbt_num_policies = max(existing_policies) + 1 if existing_policies else 1
    
    # Create PBT observer for cleanup
    observer = PbtObserver(mock_args, train_dir=train_dir)
    observer.after_init(train_dir)
    
    # Determine which policies to keep
    keep_policies = args.keep_policies if args.keep_policies is not None else existing_policies
    logger.info(f"Policies to keep: {keep_policies}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will actually be removed")
        
        # Show what would be removed
        policies_to_remove = [p for p in existing_policies if p not in keep_policies]
        if policies_to_remove:
            logger.info(f"Would remove policies: {policies_to_remove}")
        
        if args.max_age_hours:
            logger.info(f"Would remove files older than {args.max_age_hours} hours")
        
        return
    
    # Perform cleanup
    try:
        if args.cleanup_models or args.cleanup_wandb:
            # Use enhanced cleanup if additional cleanup requested
            observer.cleanup_old_policies(
                keep_policies=keep_policies,
                max_age_hours=args.max_age_hours
            )
        else:
            # Just clean up policy workspaces
            if os.path.exists(observer.pbt_workspace_dir):
                for item in os.listdir(observer.pbt_workspace_dir):
                    item_path = os.path.join(observer.pbt_workspace_dir, item)
                    if os.path.isdir(item_path) and item.startswith('policy_'):
                        try:
                            policy_idx = int(item.split('_')[1])
                            should_remove = False
                            
                            if policy_idx not in keep_policies:
                                should_remove = True
                                logger.info(f"Removing policy {policy_idx} (not in keep list)")
                            
                            elif args.max_age_hours is not None:
                                import time
                                mod_time = os.path.getmtime(item_path)
                                age_hours = (time.time() - mod_time) / 3600
                                if age_hours > args.max_age_hours:
                                    should_remove = True
                                    logger.info(f"Removing policy {policy_idx} (age: {age_hours:.1f}h)")
                            
                            if should_remove:
                                import shutil
                                shutil.rmtree(item_path)
                                logger.info(f"Removed policy workspace: {item_path}")
                                
                        except (ValueError, IndexError, OSError) as e:
                            logger.warning(f"Error processing policy directory {item}: {e}")
        
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()