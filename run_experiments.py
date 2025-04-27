# run_experiments.py - Main experiment runner for TSCFM

import os
import argparse
import pandas as pd
import numpy as np
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import TSCFM components
from tscfm import TSCFM
from data_processor import TSCFMDataProcessor
from causal_graph import CausalGraph

# Import experiment modules
from config import (
    DATASETS, COUNTERFACTUAL_METHODS, ADJUSTMENT_STRENGTHS, 
    AMPLIFICATION_FACTORS, BASE_TSCFM_CONFIG,
    get_experiment_grid, DEFAULT_OUTPUT_DIR, DATASET_PATHS,
    RANDOM_SEED
)
from utils import (
    setup_experiment_directory, save_experiment_config,
    save_experiment_results, make_experiment_id, flatten_dict
)

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
    parser = argparse.ArgumentParser(description="Run TSCFM experiments")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=None,
        choices=DATASETS,
        help="Dataset to use (if not specified, run all datasets)"
    )
    
    parser.add_argument(
        "--method", 
        type=str, 
        default=None,
        choices=COUNTERFACTUAL_METHODS,
        help="Counterfactual method to use (if not specified, run all methods)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--single_experiment", 
        action="store_true",
        help="Run only one experiment with default parameters"
    )
    
    parser.add_argument(
        "--grid_search", 
        action="store_true",
        help="Run full grid search over all parameters"
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run experiments in parallel"
    )
    
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    
    parser.add_argument(
        "--fairness_constraint", 
        type=str, 
        default=None,
        choices=["demographic_parity_difference", "equal_opportunity_difference", "equalized_odds_difference"],
        help="Fairness constraint to use (if not specified, run all constraints)"
    )

    return parser.parse_args()

# Add this function at the top of your run_experiments.py file
def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def run_single_experiment(config, output_dir=None):
    """
    Run a single TSCFM experiment with the given configuration, exactly matching
    the full TSCFM implementation from main.py
    
    Args:
        config: Dictionary with experiment configuration
        output_dir: Output directory for this experiment (if None, use default)
    
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    
    # Set global random seed to ensure reproducibility
    np.random.seed(42)
    import random
    random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass
    
    # Create experiment ID and output directory
    experiment_id = make_experiment_id(config)
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, experiment_id)
    else:
        output_dir = os.path.join(output_dir, experiment_id)
    
    # Create experiment directory structure
    dirs = setup_experiment_directory(output_dir)
    
    # Save experiment configuration
    save_experiment_config(config, output_dir)
    
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load dataset
        dataset_name = config['dataset']
        logger.info(f"Loading dataset: {dataset_name}")
        
        # For German dataset, use exact same loading procedure as in main.py
        # if dataset_name == 'german':
        #     try:
        #         # Try to import from load_german first
        #         from load_german import load_german
        #         X, y, s = load_german(mode='label_encoding')
        #         # Create feature names
        #         feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        #     except ImportError:
        #         logger.error("Could not import load_german, please ensure it's in the path")
        #         raise
        # else:
            # Handle other datasets
        dataset_path = DATASET_PATHS.get(dataset_name)
        if not dataset_path:
            raise ValueError(f"No path specified for dataset: {dataset_name}")
        
        if dataset_name == 'card_credit':
            from load_card_credit import load_card_credit
            X, y, s = load_card_credit(dataset_path, mode='label_encoding')
        elif dataset_name == 'pakdd':
            from load_pakdd import load_pkdd
            X, y, s = load_pkdd(dataset_path, mode='label_encoding')
        elif dataset_name == 'german':
            from load_german import load_german
            X, y, s = load_german(dataset_path, mode='label_encoding')
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Log dataset information
        logger.info(f"Dataset loaded: X shape={X.shape}, y shape={y.shape}, s shape={s.shape}")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
        logger.info(f"Protected attribute distribution: {np.bincount(s.astype(int))}")
        
        # Create train/test split exactly as in main.py
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=0.3, random_state=42, stratify=y
        )
        
        # Initialize data processor (for feature categorization)
        from data_processor import TSCFMDataProcessor
        data_processor = TSCFMDataProcessor(
            dataset_name=dataset_name,
            test_size=0.3,
            random_state=42
        )
        
        # Manually set the train/test split to match exactly
        data_processor.X_train = X_train
        data_processor.X_test = X_test
        data_processor.y_train = y_train
        data_processor.y_test = y_test
        data_processor.s_train = s_train
        data_processor.s_test = s_test
        data_processor.feature_names = feature_names
        data_processor.data_loaded = True
        
        # Analyze feature correlations with consistent threshold
        correlation_threshold = 0.2  # Match main.py
        feature_categories = data_processor.analyze_feature_correlations(threshold=correlation_threshold)
        
        logger.info(f"Feature categories identified:")
        logger.info(f"  Direct features: {len(feature_categories['direct'])}")
        logger.info(f"  Proxy features: {len(feature_categories['proxy'])}")
        logger.info(f"  Mediator features: {len(feature_categories['mediator'])}")
        logger.info(f"  Neutral features: {len(feature_categories['neutral'])}")
        
        # Create causal graph exactly as in main.py
        logger.info("Creating causal graph")
        from causal_graph import CausalGraph
        
        protected_attr = 'protected'  # Same as in main.py
        causal_graph = CausalGraph(protected_attribute=protected_attr)
        
        # Create DataFrame for causal discovery
        df = pd.DataFrame(X_train, columns=feature_names)
        df[protected_attr] = s_train
        df['target'] = y_train
        
        # Discover causal structure
        s_idx = df.columns.get_loc(protected_attr)
        y_idx = df.columns.get_loc('target')
        causal_graph.discover_from_data(
            df, s_idx=s_idx, outcome_idx=y_idx,
            correlation_threshold=correlation_threshold
        )
        
        # Visualize causal graph
        causal_graph_path = os.path.join(dirs['causal_graphs'], "causal_graph.png")
        try:
            causal_fig = causal_graph.visualize(figsize=(14, 10))
            # causal_fig.savefig(causal_graph_path, dpi=300, bbox_inches='tight')
            plt.close(causal_fig)
        except Exception as e:
            logger.error(f"Error visualizing causal graph: {e}")
        
        # Get base model configuration based on dataset
        if dataset_name == 'german':
            base_model_type = "random_forest"
            base_model_params = {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_split": 5,
                "class_weight": "balanced"
            }
            # Default fairness constraint for German dataset
            default_fairness_constraint = "demographic_parity_difference"
        elif dataset_name == 'card_credit':
            base_model_type = "gradient_boosting"
            base_model_params = {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.8
            }
            default_fairness_constraint = "equalized_odds_difference"
        elif dataset_name == 'pakdd':
            base_model_type = "random_forest"
            base_model_params = {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_split": 5,
                "class_weight": "balanced"
            }
            default_fairness_constraint = "demographic_parity_difference"
        else:
            # Default fallback
            base_model_type = "random_forest"
            base_model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced"
            }
            default_fairness_constraint = "demographic_parity_difference"
        
        # Override with config values if provided
        base_model_type = config.get('base_model_type', base_model_type)
        base_model_params = config.get('base_model_params', base_model_params)
        
        # Use or determine fairness constraint
        fairness_constraint = config.get('fairness_constraint', default_fairness_constraint)
        
        # Get TSCFM default values for other parameters
        # These must match the defaults in the TSCFM class constructor
        default_adjustment_strength = 0.7
        default_amplification_factor = 2.0
        
        # Override with config values if provided
        adjustment_strength = config.get('adjustment_strength', default_adjustment_strength)
        amplification_factor = config.get('amplification_factor', default_amplification_factor)
        
        # Log all parameter values for verification
        logger.info(f"TSCFM parameters:")
        logger.info(f"  base_model_type: {base_model_type}")
        logger.info(f"  base_model_params: {base_model_params}")
        logger.info(f"  counterfactual_method: {config['counterfactual_method']}")
        logger.info(f"  adjustment_strength: {adjustment_strength}")
        logger.info(f"  fairness_constraint: {fairness_constraint}")
        logger.info(f"  amplification_factor: {amplification_factor}")
        
        # Create TSCFM model with identical parameters to main.py
        from tscfm import TSCFM
        tscfm_model = TSCFM(
            base_model_type=base_model_type,
            base_model_params=base_model_params,
            counterfactual_method=config['counterfactual_method'],
            adjustment_strength=adjustment_strength,
            fairness_constraint=fairness_constraint,
            amplification_factor=amplification_factor,
            random_state=42
        )
        
        # Fit model with causal graph
        logger.info(f"Fitting TSCFM model with {config['counterfactual_method']} method")
        tscfm_model.fit(
            X_train, y_train, s_train,
            feature_names=feature_names,
            causal_graph=causal_graph
        )
        
        # Save model
        # model_path = os.path.join(dirs['models'], "tscfm_model")
        # tscfm_model.save(model_path)
        # logger.info(f"Model saved to {model_path}")
        
        # Evaluate model
        logger.info("Evaluating model")
        tscfm_dir = os.path.join(dirs['plots'], "tscfm_results")
        os.makedirs(tscfm_dir, exist_ok=True)
        
        evaluation = tscfm_model.evaluate(X_test, y_test, s_test, output_dir=tscfm_dir)
        
        # Get predictions for test data
        y_pred_fair = tscfm_model.predict(X_test, s_test)
        y_prob_fair = tscfm_model.predict_proba(X_test, s_test)[:, 1]
        
        # Get original predictions from base model
        y_pred_original = tscfm_model.base_model.predict(X_test)
        y_prob_original = tscfm_model.base_model.predict_proba(X_test)[:, 1]
        
        # Generate counterfactuals for visualization
        try:
            # Generate counterfactuals for test set
            X_cf = tscfm_model.counterfactual_generator.transform(X_test, s_test, feature_names)
            
            # Import visualization modules
            import visualization as viz
            from fairness_metrics import plot_roc_curves
            
            # Plot counterfactual distributions
            cf_dist_path = os.path.join(dirs['plots'], "counterfactual_distributions.png")
            # cf_dist_fig = viz.plot_counterfactual_distributions(
            #     X_test, X_cf, s_test, feature_names, save_path=cf_dist_path
            # )
            # cf_dist_fig = None
            # if cf_dist_fig is not None:
            #     plt.close(cf_dist_fig)
            
            # Plot embedding space
            # embedding_path = os.path.join(dirs['plots'], "embedding_space.png")
            # embedding_fig = viz.plot_embedding_space(
            #     X_test, X_cf, s_test, y_test, method='pca', save_path=embedding_path
            # )
            # if embedding_fig is not None:
            #     plt.close(embedding_fig)
            
            # # Plot outcome probabilities
            # probs_path = os.path.join(dirs['plots'], "outcome_probabilities.png")
            # probs_fig = viz.plot_outcome_probabilities(
            #     y_prob_original, y_prob_fair, s_test, save_path=probs_path
            # )
            # if probs_fig is not None:
            #     plt.close(probs_fig)
            
            # Plot ROC curves for original and fair models
            try:
                original_roc_path = os.path.join(dirs['plots'], "original_roc_curves.png")
                original_roc_fig = plot_roc_curves(
                    y_test, y_prob_original, s_test, 
                    group_names=['Group 0', 'Group 1'],
                    title="Original Model ROC Curves", 
                    save_path=original_roc_path
                )
                if original_roc_fig is not None:
                    plt.close(original_roc_fig)
                
                fair_roc_path = os.path.join(dirs['plots'], "fair_roc_curves.png")
                fair_roc_fig = plot_roc_curves(
                    y_test, y_prob_fair, s_test, 
                    group_names=['Group 0', 'Group 1'],
                    title="Fair Model ROC Curves", 
                    save_path=fair_roc_path
                )
                if fair_roc_fig is not None:
                    plt.close(fair_roc_fig)
            except Exception as e:
                logger.error(f"Error generating ROC curve visualizations: {e}")
            
            # Plot fairness metrics comparison
            comparison_path = os.path.join(dirs['plots'], "fairness_comparison.png")
            comparison_fig = viz.plot_fairness_metrics_comparison(
                evaluation['baseline']['fairness'], 
                evaluation['fair']['fairness'], 
                save_path=comparison_path
            )
            if comparison_fig is not None:
                plt.close(comparison_fig)
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        # Collect results
        results = {
            'experiment_id': experiment_id,
            'dataset': dataset_name,
            'counterfactual_method': config['counterfactual_method'],
            'fairness_constraint': config['fairness_constraint'],
            'adjustment_strength': adjustment_strength,
            'amplification_factor': amplification_factor,
            'baseline_performance': evaluation['baseline']['performance'],
            'fair_performance': evaluation['fair']['performance'],
            'baseline_fairness': evaluation['baseline']['fairness'],
            'fair_fairness': evaluation['fair']['fairness'],
            'improvement': evaluation['improvement'],
            'runtime': time.time() - start_time
        }
        
        # Save results
        save_experiment_results(results, output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print(f"EXPERIMENT RESULTS: {experiment_id}")
        print("="*50)
        print(f"\nDataset: {dataset_name}")
        print(f"Method: {config['counterfactual_method']}")
        print(f"Adjustment Strength: {adjustment_strength}")
        print(f"\nBaseline Accuracy: {results['baseline_performance']['accuracy']:.4f}")
        print(f"Fair Model Accuracy: {results['fair_performance']['accuracy']:.4f}")
        print(f"Accuracy Change: {results['fair_performance']['accuracy'] - results['baseline_performance']['accuracy']:.4f}")
        
        print("\nFairness Metrics:")
        print(f"Demographic Parity: {results['baseline_fairness']['demographic_parity_difference']:.4f} → "
              f"{results['fair_fairness']['demographic_parity_difference']:.4f} "
              f"(Improvement: {results['improvement'].get('demographic_parity_difference', 0):.1f}%)")
        
        print(f"Equal Opportunity: {results['baseline_fairness']['equal_opportunity_difference']:.4f} → "
              f"{results['fair_fairness']['equal_opportunity_difference']:.4f} "
              f"(Improvement: {results['improvement'].get('equal_opportunity_difference', 0):.1f}%)")
        print("="*50)
        
        logger.info(f"Experiment completed successfully: {experiment_id}")
        logger.info(f"Runtime: {results['runtime']:.2f} seconds")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in experiment {experiment_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())  # Print full stack trace
        
        error_info = {
            'experiment_id': experiment_id,
            'error': str(e),
            'config': config
        }
        error_path = os.path.join(output_dir, 'error.json')
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        return {'error': str(e), 'experiment_id': experiment_id}



def run_experiments_grid(args):
    """
    Run experiments for a grid of configurations
    
    Args:
        args: Command line arguments
    
    Returns:
        Dictionary with all experiment results
    """
    # Get experiment grid
    full_grid = get_experiment_grid()
    
    # Filter grid based on command line arguments
    grid = []
    for config in full_grid:
        if (args.dataset is None or config['dataset'] == args.dataset) and \
        (args.method is None or config['counterfactual_method'] == args.method) and \
        (args.fairness_constraint is None or config['fairness_constraint'] == args.fairness_constraint):
            grid.append(config)
    
    logger.info(f"Running grid search with {len(grid)} configurations")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"grid_search_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save grid configuration
    grid_config = {
        'timestamp': timestamp,
        'num_experiments': len(grid),
        'dataset_filter': args.dataset,
        'method_filter': args.method
    }
    with open(os.path.join(output_dir, 'grid_config.json'), 'w') as f:
        json.dump(convert_numpy_types(grid_config), f, indent=2)
    
    # Run experiments - check if parallel or sequential
    if args.parallel:
        from joblib import Parallel, delayed
        
        logger.info(f"Running experiments in parallel with {args.n_jobs} jobs")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_single_experiment)(config, output_dir) for config in tqdm(grid)
        )
    else:
        # Run sequentially with progress bar
        results = []
        for config in tqdm(grid, desc="Running experiments"):
            result = run_single_experiment(config, output_dir)
            results.append(result)
    
    # Compile all results
    all_results = {}
    for result in results:
        if 'experiment_id' in result and 'error' not in result:
            all_results[result['experiment_id']] = result
    
    # Save combined results
    combined_path = os.path.join(output_dir, 'all_results.json')
    with open(combined_path, 'w') as f:
        converted_results = convert_numpy_types(all_results)
        json.dump(converted_results, f, indent=2)
    
    # Also save as CSV for easier analysis
    try:
        # Flatten nested dictionaries
        flat_results = []
        for exp_id, result in all_results.items():
            flat_result = flatten_dict(result)
            flat_results.append(flat_result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(flat_results)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'all_results.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Combined results saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error saving combined results as CSV: {e}")
    
    logger.info(f"Grid search completed with {len(all_results)} successful experiments")
    logger.info(f"Results saved to {output_dir}")
    
    return all_results

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Starting TSCFM experiments")
    
    if args.single_experiment:
        # Run single experiment with default parameters
        default_config = {
            'dataset': args.dataset or 'german',
            'counterfactual_method': args.method or 'structural_equation',
            'adjustment_strength': 0.7,
            'amplification_factor': 2.0
        }
        default_config.update(BASE_TSCFM_CONFIG)
        
        logger.info("Running single experiment with default parameters")
        result = run_single_experiment(default_config)
        
        # Print summary
        if 'error' not in result:
            logger.info("\n=== Experiment Results ===")
            logger.info(f"Dataset: {result['dataset']}")
            logger.info(f"Method: {result['counterfactual_method']}")
            logger.info(f"Baseline Accuracy: {result['baseline_performance']['accuracy']:.4f}")
            logger.info(f"Fair Model Accuracy: {result['fair_performance']['accuracy']:.4f}")
            logger.info(f"Baseline SP: {result['baseline_fairness']['demographic_parity_difference']:.4f}")
            logger.info(f"Fair Model SP: {result['fair_fairness']['demographic_parity_difference']:.4f}")
            logger.info(f"SP Improvement: {result['improvement'].get('demographic_parity_difference', 0):.1f}%")
            logger.info("==========================")
        
        return result
    
    elif args.grid_search:
        # Run grid search
        return run_experiments_grid(args)
    
    else:
        # Default behavior if no specific mode is selected
        logger.info("No experiment mode specified. Run with --single_experiment or --grid_search")
        logger.info("Using default: single experiment")
        
        default_config = {
            'dataset': args.dataset or 'german',
            'counterfactual_method': args.method or 'structural_equation',
            'adjustment_strength': 0.7,
            'amplification_factor': 2.0
        }
        default_config.update(BASE_TSCFM_CONFIG)
        
        return run_single_experiment(default_config)


# import sys

# def main():
#     """Main function."""

#     # Simulate CLI arguments for debugging or scripted execution
#     sys.argv = [
#         "run_tscfm_experiments.py",  # dummy script name
#         # "--mode", "all",
#         "--dataset", "german",
#         "--method", "structural_equation",
#         "--grid_search"
#     ]

#     args = parse_args()
#     logger.info("Starting TSCFM experiments")

#     if args.single_experiment:
#         # Run single experiment with default parameters
#         default_config = {
#             'dataset': args.dataset or 'german',
#             'counterfactual_method': args.method or 'structural_equation',
#             'adjustment_strength': 0.7,
#             'amplification_factor': 2.0
#         }
#         default_config.update(BASE_TSCFM_CONFIG)

#         logger.info("Running single experiment with default parameters")
#         result = run_single_experiment(default_config)

#         # Print summary
#         if 'error' not in result:
#             logger.info("\n=== Experiment Results ===")
#             logger.info(f"Dataset: {result['dataset']}")
#             logger.info(f"Method: {result['counterfactual_method']}")
#             logger.info(f"Baseline Accuracy: {result['baseline_performance']['accuracy']:.4f}")
#             logger.info(f"Fair Model Accuracy: {result['fair_performance']['accuracy']:.4f}")
#             logger.info(f"Baseline SP: {result['baseline_fairness']['demographic_parity_difference']:.4f}")
#             logger.info(f"Fair Model SP: {result['fair_fairness']['demographic_parity_difference']:.4f}")
#             logger.info(f"SP Improvement: {result['improvement'].get('demographic_parity_difference', 0):.1f}%")
#             logger.info("==========================")

#         return result

#     elif args.grid_search:
#         return run_experiments_grid(args)

#     else:
#         logger.info("No experiment mode specified. Run with --single_experiment or --grid_search")
#         logger.info("Using default: single experiment")

#         default_config = {
#             'dataset': args.dataset or 'german',
#             'counterfactual_method': args.method or 'structural_equation',
#             'adjustment_strength': 0.7,
#             'amplification_factor': 2.0
#         }
#         default_config.update(BASE_TSCFM_CONFIG)

#         return run_single_experiment(default_config)


if __name__ == "__main__":
    main()