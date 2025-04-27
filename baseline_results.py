import pandas as pd
import numpy as np
from config import BENCHMARK_RESULTS, METRIC_MAPPING, DATASETS

def get_baseline_results(dataset_name):
    """
    Get baseline results for a specific dataset
    
    Args:
        dataset_name: Name of the dataset ('german', 'card_credit', or 'pakdd')
    
    Returns:
        DataFrame with baseline results
    """
    if dataset_name not in BENCHMARK_RESULTS:
        raise ValueError(f"No benchmark results available for dataset: {dataset_name}")
    
    return BENCHMARK_RESULTS[dataset_name].copy()

def get_best_baseline_models(dataset_name, metric='Acc', fairness_metric='SP', n_models=3):
    """
    Get the best baseline models for a specific dataset based on a performance metric
    
    Args:
        dataset_name: Name of the dataset
        metric: Performance metric to use for ranking
        fairness_metric: Fairness metric to consider
        n_models: Number of top models to return
    
    Returns:
        DataFrame with top models
    """
    # Get baseline results
    results = get_baseline_results(dataset_name)
    
    # Sort by performance metric (descending)
    sorted_results = results.sort_values(metric, ascending=False)
    
    # Return top n models
    return sorted_results.head(n_models)

def get_fairest_baseline_models(dataset_name, fairness_metric='SP', n_models=3):
    """
    Get the fairest baseline models for a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        fairness_metric: Fairness metric to use for ranking
        n_models: Number of top models to return
    
    Returns:
        DataFrame with fairest models
    """
    # Get baseline results
    results = get_baseline_results(dataset_name)
    
    # For fairness metrics, lower absolute value is better
    results['abs_fairness'] = results[fairness_metric].abs()
    
    # Sort by fairness (ascending)
    sorted_results = results.sort_values('abs_fairness')
    
    # Return top n models (dropping the temporary column)
    return sorted_results.drop(columns=['abs_fairness']).head(n_models)

def compare_with_baselines(tscfm_results, dataset_name, metrics=None):
    """
    Compare TSCFM results with baseline models
    
    Args:
        tscfm_results: Dictionary with TSCFM results
        dataset_name: Name of the dataset
        metrics: List of metrics to compare (if None, use all available)
    
    Returns:
        DataFrame with comparison results
    """
    # Get baseline results
    baseline_df = get_baseline_results(dataset_name)
    
    # Determine metrics to compare
    if metrics is None:
        # Use all metrics that are in both baseline results and TSCFM results
        metrics = [col for col in baseline_df.columns if col in tscfm_results and col != 'Model']
    
    # Convert TSCFM results to a DataFrame row
    tscfm_row = pd.DataFrame([tscfm_results])
    
    # Create a descriptive model name based on configuration
    if 'counterfactual_method' in tscfm_results and 'fairness_constraint' in tscfm_results:
        method = tscfm_results['counterfactual_method']
        constraint = tscfm_results['fairness_constraint']
        tscfm_row['Model'] = f"TSCFM_{method}_{constraint}"
    else:
        # Fallback to generic name if config details aren't available
        tscfm_row['Model'] = 'TSCFM'
    
    # Select only the metrics to compare
    baseline_df = baseline_df[['Model'] + metrics]
    tscfm_row = tscfm_row[['Model'] + metrics]
    
    # Combine baseline and TSCFM results
    combined_df = pd.concat([baseline_df, tscfm_row], ignore_index=True)
    
    return combined_df

def calculate_improvement(tscfm_results, dataset_name, comparison_models=None):
    """
    Calculate improvement of TSCFM over baseline models
    
    Args:
        tscfm_results: Dictionary with TSCFM results
        dataset_name: Name of the dataset
        comparison_models: List of baseline models to compare with (if None, use all)
    
    Returns:
        Dictionary with improvement percentages
    """
    # Get baseline results
    baseline_df = get_baseline_results(dataset_name)
    
    # Filter baseline models if specified
    if comparison_models is not None:
        baseline_df = baseline_df[baseline_df['Model'].isin(comparison_models)]
    
    # Initialize results dictionary
    improvements = {}
    
    # Calculate improvements for each metric
    for metric, baseline_col in METRIC_MAPPING.items():
        if baseline_col in baseline_df.columns and metric in tscfm_results:
            # For fairness metrics (except accuracy metrics), lower absolute value is better
            if baseline_col not in ['BA', 'Acc']:
                # Calculate mean absolute value for baseline models
                baseline_mean = baseline_df[baseline_col].abs().mean()
                # Calculate TSCFM absolute value
                tscfm_value = abs(tscfm_results[metric])
                # Calculate improvement percentage (negative means worse)
                if baseline_mean > 0:
                    imp_percent = (baseline_mean - tscfm_value) / baseline_mean * 100
                else:
                    imp_percent = 0.0
            else:
                # For accuracy metrics, higher is better
                baseline_mean = baseline_df[baseline_col].mean()
                tscfm_value = tscfm_results[metric]
                # Calculate improvement percentage (negative means worse)
                if baseline_mean > 0:
                    imp_percent = (tscfm_value - baseline_mean) / baseline_mean * 100
                else:
                    imp_percent = 0.0
            
            improvements[metric] = {
                'baseline_mean': float(baseline_mean),
                'tscfm_value': float(tscfm_value),
                'improvement_percent': float(imp_percent)
            }
    
    return improvements

def get_pareto_frontier(dataset_name, perf_metric='Acc', fair_metric='SP'):
    """
    Get Pareto frontier of baseline models (non-dominated in terms of performance and fairness)
    
    Args:
        dataset_name: Name of the dataset
        perf_metric: Performance metric
        fair_metric: Fairness metric
    
    Returns:
        DataFrame with Pareto-optimal models
    """
    # Get baseline results
    results = get_baseline_results(dataset_name)
    
    # For fairness metrics, lower absolute value is better
    results['abs_fairness'] = results[fair_metric].abs()
    
    # Initialize list of Pareto-optimal models
    pareto_models = []
    
    # Iterate through all models
    for idx, row in results.iterrows():
        # Check if model is dominated by any other model
        is_dominated = False
        
        for idx2, row2 in results.iterrows():
            if idx != idx2:
                # Model 2 dominates Model 1 if it has better performance and fairness
                if (row2[perf_metric] >= row[perf_metric] and 
                    row2['abs_fairness'] <= row['abs_fairness'] and
                    (row2[perf_metric] > row[perf_metric] or 
                     row2['abs_fairness'] < row['abs_fairness'])):
                    is_dominated = True
                    break
        
        # If model is not dominated, add to Pareto frontier
        if not is_dominated:
            pareto_models.append(row)
    
    # Convert list to DataFrame and sort by performance
    pareto_df = pd.DataFrame(pareto_models).sort_values(perf_metric, ascending=False)
    
    # Drop temporary column
    return pareto_df.drop(columns=['abs_fairness'])

def create_benchmark_summary(dataset_name=None):
    """
    Create a summary of benchmark results
    
    Args:
        dataset_name: Name of dataset (if None, summarize all datasets)
    
    Returns:
        Dictionary with benchmark summaries
    """
    summary = {}
    
    # Determine which datasets to summarize
    datasets_to_summarize = [dataset_name] if dataset_name else DATASETS
    
    for ds in datasets_to_summarize:
        if ds in BENCHMARK_RESULTS:
            results = get_baseline_results(ds)
            
            # Calculate statistics for each metric
            ds_summary = {}
            for metric in results.columns:
                if metric != 'Model':
                    metric_values = results[metric].dropna()
                    if len(metric_values) > 0:
                        ds_summary[metric] = {
                            'mean': float(metric_values.mean()),
                            'std': float(metric_values.std()),
                            'min': float(metric_values.min()),
                            'max': float(metric_values.max()),
                            'median': float(metric_values.median())
                        }
            
            summary[ds] = ds_summary
    
    return summary