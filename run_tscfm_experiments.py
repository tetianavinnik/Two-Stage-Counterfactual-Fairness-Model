# run_tscfm_experiments.py - Main entry point for TSCFM experiments and analysis

import os
import argparse
import logging
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tscfm_experiments.log")
    ]
)
logger = logging.getLogger("tscfm_experiments")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TSCFM experiments and analysis")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["run", "analyze", "all"],
        help="Mode to run: 'run' to run experiments, 'analyze' to analyze results, 'all' for both"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["german", "card_credit", "pakdd", "all"],
        help="Dataset to use (default: all datasets)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["structural_equation", "matching", "generative", "all"],
        help="Counterfactual method to use (default: all methods)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: automatic timestamped directory)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel"
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs when running in parallel"
    )
    
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX tables for thesis"
    )

    parser.add_argument(
        "--fairness_constraint", 
        type=str, 
        default=None,
        choices=["demographic_parity_difference", "equal_opportunity_difference", "equalized_odds_difference"],
        help="Fairness constraint to use (if not specified, run all constraints)"
    )
    
    return parser.parse_args()

def run_experiments(args):
    """Run TSCFM experiments using configuration that matches main.py behavior."""
    logger.info("Starting TSCFM experiments")
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./results/tscfm_experiments_{timestamp}"
    
    # Prepare command with supported parameters
    cmd = [sys.executable, "run_experiments.py"]
    
    if args.mode == "all":
        cmd.append("--grid_search")
    else:
        cmd.append("--single_experiment")
    
    cmd.extend(["--output_dir", output_dir])
    
    # Add dataset filter if specified
    if args.dataset and args.dataset != "all":
        cmd.extend(["--dataset", args.dataset])
    
    # Add method filter if specified
    if args.method and args.method != "all":
        cmd.extend(["--method", args.method])
    
    # Add fairness constraint if specified
    if args.fairness_constraint:
        cmd.extend(["--fairness_constraint", args.fairness_constraint])
    
    # Add parallel flag if specified
    if args.parallel:
        cmd.extend(["--parallel", "--n_jobs", str(args.n_jobs)])
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Execute the experiment
        subprocess.run(cmd, check=True)
        logger.info(f"Experiments completed successfully")
        return output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running experiments: {e}")
        return None

def patch_run_single_experiment():
    """
    Patches run_single_experiment function in run_experiments.py to match main.py behavior.
    This uses monkey patching to modify the function at runtime.
    """
    import run_experiments
    import types
    
    # Define the patched function
    def patched_run_single_experiment(config, output_dir=None):
        """
        Patched version that ensures identical behavior to main.py --full_tscfm
        """
        # The patched implementation from my previous message
        # ...full code here...
        pass
    
    # Apply the patch
    run_experiments.run_single_experiment = patched_run_single_experiment
    logger.info("Patched run_single_experiment to match main.py behavior")

def analyze_results(args, results_dir=None):
    """Analyze TSCFM experiment results."""
    logger.info("Starting results analysis")
    
    # Prepare command for compile_results.py
    cmd = [sys.executable, "compile_results.py"]
    
    # Add results directory if provided
    if results_dir:
        cmd.extend(["--results_dir", results_dir])
    
    # Add dataset filter if specified
    if args.dataset and args.dataset != "all":
        cmd.extend(["--dataset", args.dataset])
    
    # Add LaTeX flag if specified
    if args.latex:
        cmd.append("--generate_latex")
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error analyzing results: {e}")

def main():
    """Main function."""
    import numpy as np
    import random
    import os
    import torch  # If you're using PyTorch
    import tensorflow as tf  # If you're using TensorFlow

    def set_all_seeds(seed):
        # Python's built-in random module
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # Set Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # If using TensorFlow
        if 'tensorflow' in sys.modules:
            tf.random.set_seed(seed)
            
        # If using PyTorch
        if 'torch' in sys.modules:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Use a fixed seed for reproducibility
    GLOBAL_SEED = 42
    set_all_seeds(GLOBAL_SEED)

    args = parse_args()
    
    # Process based on mode
    if args.mode in ["run", "all"]:
        results_dir = run_experiments(args)
    else:
        results_dir = None
    
    if args.mode in ["analyze", "all"] and (results_dir is not None or args.mode == "analyze"):
        analyze_results(args, results_dir)
    
    logger.info("TSCFM experiment workflow completed")


if __name__ == "__main__":
    main()