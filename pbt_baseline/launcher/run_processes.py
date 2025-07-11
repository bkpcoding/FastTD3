"""Local process launcher for PBT experiments."""

import os
import subprocess
import time
import signal
from typing import List, Dict


def add_os_parallelism_args(parser):
    """Add arguments for OS parallelism."""
    parser.add_argument("--max_parallel", default=4, type=int, help="Maximum number of parallel processes")
    parser.add_argument("--experiments_per_gpu", default=4, type=int, help="Number of experiments per GPU")
    parser.add_argument("--gpus", default="0", type=str, help="Comma-separated list of GPU IDs to use")
    return parser


class ProcessManager:
    """Manage multiple training processes."""
    
    def __init__(self, max_parallel: int = 4):
        self.max_parallel = max_parallel
        self.running_processes = []
        self.completed_processes = []
        self.failed_processes = []
    
    def start_process(self, cmd: str, name: str, cwd: str = None, env: dict = None):
        """Start a new process."""
        print(f"Starting experiment: {name}")
        print(f"Command: {cmd}")
        
        # Create experiment directory
        if cwd:
            os.makedirs(cwd, exist_ok=True)
        
        # Set up environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        # Add current directory to PYTHONPATH to ensure module can be found
        current_dir = os.getcwd()
        if 'PYTHONPATH' in process_env:
            process_env['PYTHONPATH'] = f"{current_dir}:{process_env['PYTHONPATH']}"
        else:
            process_env['PYTHONPATH'] = current_dir
        
        # Start process
        try:
            # For debugging, let's not capture output so we can see errors
            # Run from current directory instead of experiment directory to keep module path
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=current_dir,  # Use main project directory
                env=process_env,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.STDOUT,
                universal_newlines=True,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.running_processes.append({
                'process': process,
                'name': name,
                'cmd': cmd,
                'start_time': time.time(),
            })
            
            return True
            
        except Exception as e:
            print(f"Failed to start process {name}: {e}")
            return False
    
    def check_processes(self):
        """Check status of running processes."""
        still_running = []
        
        for proc_info in self.running_processes:
            process = proc_info['process']
            return_code = process.poll()
            
            if return_code is None:
                # Still running
                still_running.append(proc_info)
            elif return_code == 0:
                # Completed successfully
                print(f"Experiment {proc_info['name']} completed successfully")
                self.completed_processes.append(proc_info)
            else:
                # Failed
                print(f"Experiment {proc_info['name']} failed with return code {return_code}")
                self.failed_processes.append(proc_info)
        
        self.running_processes = still_running
    
    def wait_for_slot(self):
        """Wait until there's a free slot for new process."""
        while len(self.running_processes) >= self.max_parallel:
            time.sleep(1)
            self.check_processes()
    
    def wait_for_all(self):
        """Wait for all processes to complete."""
        while self.running_processes:
            time.sleep(1)
            self.check_processes()
    
    def terminate_all(self):
        """Terminate all running processes."""
        for proc_info in self.running_processes:
            try:
                # Kill entire process group
                os.killpg(os.getpgid(proc_info['process'].pid), signal.SIGTERM)
            except Exception as e:
                print(f"Failed to terminate process {proc_info['name']}: {e}")
        
        # Wait a bit then force kill if needed
        time.sleep(5)
        for proc_info in self.running_processes:
            try:
                if proc_info['process'].poll() is None:
                    os.killpg(os.getpgid(proc_info['process'].pid), signal.SIGKILL)
            except Exception:
                pass
    
    def print_summary(self):
        """Print summary of all processes."""
        total = len(self.completed_processes) + len(self.failed_processes)
        print(f"\n=== Process Summary ===")
        print(f"Total experiments: {total}")
        print(f"Completed successfully: {len(self.completed_processes)}")
        print(f"Failed: {len(self.failed_processes)}")
        
        if self.failed_processes:
            print("\nFailed experiments:")
            for proc_info in self.failed_processes:
                print(f"  - {proc_info['name']}")


def run(run_description, launcher_cfg):
    """Run experiments using local processes."""
    print(f"Starting run: {run_description.run_name}")
    
    # Generate all experiment commands
    experiments = run_description.generate_experiments(launcher_cfg.train_dir)
    print(f"Generated {len(experiments)} experiments")
    
    # Parse GPU configuration
    gpu_ids = [int(gpu.strip()) for gpu in launcher_cfg.gpus.split(",") if gpu.strip()]
    
    # Create process manager
    manager = ProcessManager(launcher_cfg.max_parallel)
    
    try:
        for i, experiment in enumerate(experiments):
            # Wait for available slot
            manager.wait_for_slot()
            
            # Assign GPU
            gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else 0
            env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
            
            # Add device rank to command
            # cmd = experiment['cmd'] + f" device_rank 0"  # Always 0 since we set CUDA_VISIBLE_DEVICES
            cmd = experiment['cmd']
            # Start process
            success = manager.start_process(
                cmd=cmd,
                name=experiment['name'],
                cwd=experiment['dir'],
                env=env
            )
            
            if not success:
                print(f"Failed to start experiment {experiment['name']}")
                continue
            
            # Pause between starting processes
            if hasattr(launcher_cfg, 'pause_between'):
                time.sleep(launcher_cfg.pause_between)
        
        # Wait for all processes to complete
        print("Waiting for all experiments to complete...")
        manager.wait_for_all()
        
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating all processes...")
        manager.terminate_all()
    
    finally:
        manager.print_summary()
    
    return len(manager.failed_processes) == 0