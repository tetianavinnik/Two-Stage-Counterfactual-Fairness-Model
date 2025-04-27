# utils.py - Helper functions for TSCFM experiments

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import logging

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

def setup_experiment_directory(output_dir):
    """
    Create necessary directories for experiment outputs
    
    Args:
        output_dir: Base directory for experiment outputs
    
    Returns:
        Dictionary with paths for different output types
    """
    # Create main directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'results': os.path.join(output_dir, 'results'),
        'models': os.path.join(output_dir, 'models'),
        'plots': os.path.join(output_dir, 'plots'),
        'causal_graphs': os.path.join(output_dir, 'causal_graphs'),
        'feature_analysis': os.path.join(output_dir, 'feature_analysis'),
        'counterfactuals': os.path.join(output_dir, 'counterfactuals'),
        'roc_curves': os.path.join(output_dir, 'roc_curves')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    return subdirs

def save_experiment_config(config, output_dir):
    """
    Save experiment configuration to a JSON file
    
    Args:
        config: Dictionary with experiment configuration
        output_dir: Directory to save the config
    """
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved experiment config to {config_path}")

def save_experiment_results(results, output_dir, filename='results.json'):
    """
    Save experiment results to a JSON file
    
    Args:
        results: Dictionary with experiment results
        output_dir: Directory to save the results
        filename: Name of the results file
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj
    
    # Recursively convert dictionary values
    def convert_dict(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = convert_dict(v)
            else:
                result[k] = convert_to_serializable(v)
        return result
    
    # Convert results dictionary
    serializable_results = convert_dict(results)
    
    # Save to file
    results_path = os.path.join(output_dir, filename)
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved experiment results to {results_path}")
    
    # Also save as CSV for easier analysis
    if isinstance(results, dict) and not isinstance(next(iter(results.values()), None), dict):
        # If results is a flat dictionary, convert to DataFrame
        df = pd.DataFrame([results])
        csv_path = os.path.join(output_dir, filename.replace('.json', '.csv'))
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved experiment results as CSV to {csv_path}")

def load_experiment_results(output_dir, filename='results.json'):
    """
    Load experiment results from a JSON file
    
    Args:
        output_dir: Directory containing the results
        filename: Name of the results file
    
    Returns:
        Dictionary with experiment results
    """
    results_path = os.path.join(output_dir, filename)
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def calculate_performance_metrics(y_true, y_pred, y_prob):
    """
    Calculate standard performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
    
    Returns:
        Dictionary with performance metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = np.nan
    
    return metrics

def select_top_features(feature_importances, feature_names, top_n=10):
    """
    Select top N most important features
    
    Args:
        feature_importances: Array of feature importance values
        feature_names: List of feature names
        top_n: Number of top features to select
    
    Returns:
        List of (feature_name, importance) tuples for top features
    """
    # Create feature importance pairs
    importance_pairs = list(zip(feature_names, feature_importances))
    
    # Sort by importance (descending)
    sorted_pairs = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
    
    # Return top N
    return sorted_pairs[:top_n]

def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary
    
    Args:
        d: Nested dictionary
        parent_key: Parent key for recursive calls
        sep: Separator between keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def make_experiment_id(config):
    """
    Create a unique experiment ID from config
    """
    return (f"{config['dataset']}_{config['counterfactual_method']}_"
            f"{config['fairness_constraint']}_"  # Add this line
            f"adj{config['adjustment_strength']}_"
            f"amp{config['amplification_factor']}")


def make_model_name(config):
    """
    Create a readable model name from config parameters.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        A descriptive model name string
    """
    # Method abbreviations
    method_abbr = {
        "structural_equation": "S",
        "matching": "M",
        "generative": "G"
    }
    
    # Constraint abbreviations
    constraint_abbr = {
        "demographic_parity_difference": "DP",
        "equal_opportunity_difference": "EO",
        "equalized_odds_difference": "EOd"
    }
    
    # Get method abbreviation
    method = method_abbr.get(config['counterfactual_method'], "X")
    
    # Get constraint abbreviation
    constraint = constraint_abbr.get(config['fairness_constraint'], "X")
    
    # Format parameters with single digit precision
    adj = f"{config['adjustment_strength']:.1f}"
    amp = f"{config['amplification_factor']:.1f}"
    
    # Create model name in format: Method-Constraint-AdjStr-Amp
    # Example: S-DP-0.7-2.0
    model_name = f"{method}-{constraint}-{adj}-{amp}"
    
    return model_name