import os
import argparse
import pandas as pd
import numpy as np
import json
import logging
import time
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import experiment modules
from config import (
    DATASETS, COUNTERFACTUAL_METHODS, METRIC_MAPPING, PRIMARY_FAIRNESS_METRICS, 
    METRIC_DISPLAY_NAMES, DEFAULT_OUTPUT_DIR
)
from utils import (
    flatten_dict
)

# Import visualization modules
from performance import (
    plot_fairness_performance_scatter,
    plot_hyperparameter_sensitivity, plot_counterfactual_method_comparison, plot_trade_off_scatter,
    plot_fairness_constraint_comparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tscfm_results.log")
    ]
)
logger = logging.getLogger("tscfm_results")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compile TSCFM experiment results")
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=None,
        help="Directory containing experiment results (if None, use latest in default dir)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save compiled results (if None, use results_dir/compiled)"
    )
    
    parser.add_argument(
        "--generate_latex", 
        action="store_true",
        help="Generate LaTeX tables for thesis"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=None,
        choices=DATASETS,
        help="Dataset to focus on (if None, process all datasets)"
    )
    
    return parser.parse_args()

def find_latest_results_dir():
    """Find the most recent results directory."""
    all_dirs = glob(os.path.join(DEFAULT_OUTPUT_DIR, "*"))
    # Filter out non-directories
    all_dirs = [d for d in all_dirs if os.path.isdir(d)]
    if not all_dirs:
        raise ValueError(f"No experiment directories found in {DEFAULT_OUTPUT_DIR}")
    
    # Sort by modification time (most recent first)
    all_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Return the most recent directory
    return all_dirs[0]

def compile_experiment_results(experiments_dir, output_dir, dataset_filter=None):
    """
    Compile results from a directory of experiments
    
    Args:
        experiments_dir: Directory containing experiment results
        output_dir: Directory to save compiled results
        dataset_filter: Dataset to filter results (if None, process all)
    
    Returns:
        DataFrame with compiled results
    """
    logger.info(f"Compiling results from {experiments_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result files
    result_files = glob(os.path.join(experiments_dir, "**", "results.json"), recursive=True)
    
    if not result_files:
        logger.warning(f"No result files found in {experiments_dir}")
        return None
    
    logger.info(f"Found {len(result_files)} result files")
    
    # Load and compile results
    compiled_results = []
    
    for result_file in result_files:
        try:
            # Load results
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # Filter by dataset if specified
            if dataset_filter and results.get('dataset') != dataset_filter:
                continue
            
            # Flatten nested dictionary
            flat_results = flatten_dict(results)
            
            # Add to compiled results
            compiled_results.append(flat_results)
            
        except Exception as e:
            logger.error(f"Error processing {result_file}: {e}")
    
    if not compiled_results:
        logger.warning("No valid results found after filtering")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(compiled_results)

    csv_path = os.path.join(output_dir, "compiled_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved compiled results to {csv_path}")
    
    return df

def generate_combined_visualizations(results_df, output_dir, dataset=None):
    """
    Generate visualizations from compiled results
    
    Args:
        results_df: DataFrame with compiled results
        output_dir: Directory to save visualizations
        dataset: Specific dataset to focus on (if None, handle all)
    """
    logger.info("Generating combined visualizations")
    
    # Create visualization directories
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    performance_dir = os.path.join(viz_dir, "performance")
    fairness_dir = os.path.join(viz_dir, "fairness")
    methods_dir = os.path.join(viz_dir, "methods")
    params_dir = os.path.join(viz_dir, "parameters")
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(fairness_dir, exist_ok=True)
    os.makedirs(methods_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    
    # Determine datasets to process
    if dataset:
        datasets = [dataset]
    else:
        datasets = results_df['dataset'].unique()
    
    # Process each dataset
    for dataset in datasets:
        logger.info(f"Processing visualizations for dataset: {dataset}")
        
        # Filter results for this dataset
        ds_results = results_df[results_df['dataset'] == dataset]
        
        # 1. Counterfactual method comparison
        try:
            method_path = os.path.join(methods_dir, f"{dataset}_method_comparison.png")
            plot_counterfactual_method_comparison(ds_results, output_path=method_path)
            logger.info(f"Generated method comparison plot for {dataset}")
        except Exception as e:
            logger.error(f"Error generating method comparison for {dataset}: {e}")
        
        # 2. Parameter sensitivity plots
        for param in ['adjustment_strength', 'amplification_factor']:
            try:
                # For fairness metrics - with constraint separation and boxplots
                fairness_metrics = ['fair_fairness_demographic_parity_difference', 
                                'fair_fairness_equal_opportunity_difference',
                                'fair_fairness_equalized_odds_difference']
                
                for metric in fairness_metrics:
                    fairness_path = os.path.join(params_dir, f"{dataset}_{param}_{metric}_by_constraint.png")
                    plot_hyperparameter_sensitivity(
                        ds_results, param, metric, 
                        output_path=fairness_path,
                        use_boxplot=True,  # Use boxplots
                        separate_by_constraint=True  # Separate by fairness constraint
                    )
                
                # For accuracy - standard boxplot without constraint separation
                acc_path = os.path.join(params_dir, f"{dataset}_{param}_accuracy.png")
                plot_hyperparameter_sensitivity(
                    ds_results, param, 'fair_performance_accuracy', 
                    output_path=acc_path,
                    use_boxplot=True,  # Use boxplots
                    separate_by_constraint=False  # No need to separate by constraint for accuracy
                )
                
                logger.info(f"Generated parameter sensitivity plots for {param} on {dataset}")
            except Exception as e:
                logger.error(f"Error generating parameter plots for {param} on {dataset}: {e}")
        
        # 3. Model comparison with baselines
        try:
            for dataset in datasets:
                logger.info(f"Processing fairness comparison for dataset: {dataset}")
                
                # Filter results for this dataset
                ds_results = results_df[results_df['dataset'] == dataset]
                
                # Get baseline results for comparison
                from baseline_results import get_baseline_results
                baseline_df = get_baseline_results(dataset)
                
                # Create combined comparison DataFrame with baseline models
                comparison_df = baseline_df.copy()

                # Add each TSCFM variant to the comparison
                for idx, row in ds_results.iterrows():
                    # Import the make_model_name function
                    from utils import make_model_name
                    
                    # Create model name using make_model_name function
                    config = {
                        'counterfactual_method': row['counterfactual_method'],
                        'fairness_constraint': row['fairness_constraint'],
                        'adjustment_strength': row['adjustment_strength'],
                        'amplification_factor': row['amplification_factor']
                    }
                    model_name = make_model_name(config)
                    
                    # Map results to benchmark format
                    benchmark_format = {
                        'BA': row['fair_performance_roc_auc'],  # Use ROC AUC instead of balanced accuracy
                        'Acc': row['fair_performance_accuracy'],
                        'SP': row['fair_fairness_demographic_parity_difference'],
                        'EO': row['fair_fairness_equal_opportunity_difference'],
                        'EOd': row['fair_fairness_equalized_odds_difference'],
                        'PP': row.get('fair_fairness_predictive_parity_difference', 0),
                        'PE': row.get('fair_fairness_predictive_equality_difference', 0),
                        'TE': row.get('fair_fairness_treatment_equality_difference', 0),
                        'ABROCA': row.get('fair_fairness_abroca', 0)
                    }
                    
                    # Add to comparison DataFrame
                    tscfm_row = pd.DataFrame([{'Model': model_name} | benchmark_format])
                    comparison_df = pd.concat([comparison_df, tscfm_row], ignore_index=True)
                
                comparison_path = os.path.join(performance_dir, f"{dataset}_baseline_comparison.csv")
                comparison_df.to_csv(comparison_path, index=False)
                
                # Create fairness-performance scatter plots for each fairness metric
                for fair_metric in PRIMARY_FAIRNESS_METRICS:
                    scatter_path = os.path.join(fairness_dir, f"{dataset}_fairness_scatter_{fair_metric}.png")
                    plot_fairness_performance_scatter(
                        comparison_df, 
                        fairness_metric=fair_metric, 
                        output_path=scatter_path
                    )
                
                logger.info(f"Generated baseline comparison visualizations for {dataset}")
        except Exception as e:
            logger.error(f"Error generating baseline comparison: {e}")
    
    # Generate combined performance comparison across datasets
    try:
        # Get best model for each dataset
        best_models = []
        
        for dataset in datasets:
            ds_results = results_df[results_df['dataset'] == dataset]
            if not ds_results.empty:
                best_model = ds_results.iloc[ds_results['improvement_demographic_parity_difference'].idxmax()]
                best_model_data = {
                    'Dataset': dataset,
                    'Method': best_model['counterfactual_method'],
                    'Accuracy': best_model['fair_performance_accuracy'],
                    'DP Diff': best_model['fair_fairness_demographic_parity_difference'],
                    'EO Diff': best_model['fair_fairness_equal_opportunity_difference'],
                    'Acc Improvement': best_model['fair_performance_accuracy'] - best_model['baseline_performance_accuracy'],
                    'DP Improvement': best_model['improvement_demographic_parity_difference'],
                    'EO Improvement': best_model['improvement_equal_opportunity_difference']
                }
                best_models.append(best_model_data)
        
        best_df = pd.DataFrame(best_models)

        summary_path = os.path.join(output_dir, "best_models_summary.csv")
        best_df.to_csv(summary_path, index=False)

        if len(best_df) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            ax1.bar(best_df['Dataset'], best_df['Accuracy'], color='teal')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy by Dataset')
  
            for i, v in enumerate(best_df['Accuracy']):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
  
            x = np.arange(len(best_df))
            width = 0.35
            
            ax2.bar(x - width/2, best_df['DP Improvement'], width, 
                  label='DP Improvement')
            ax2.bar(x + width/2, best_df['EO Improvement'], width, 
                  label='EO Improvement')
            
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Fairness Improvement by Dataset')
            ax2.set_xticks(x)
            ax2.set_xticklabels(best_df['Dataset'])
            ax2.legend()
  
            for i, v in enumerate(best_df['DP Improvement']):
                ax2.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
            
            for i, v in enumerate(best_df['EO Improvement']):
                ax2.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
            
            plt.suptitle('Cross-Dataset Performance Summary', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            multi_path = os.path.join(viz_dir, "cross_dataset_comparison.png")
            plt.savefig(multi_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated cross-dataset comparison plot")
    except Exception as e:
        logger.error(f"Error generating cross-dataset comparison: {e}")
    
    logger.info("Completed combined visualizations")

def generate_latex_tables(results_df, baseline_results, output_dir):
    """
    Generate LaTeX tables
    
    Args:
        results_df: DataFrame with compiled results
        baseline_results: Dictionary with baseline results
        output_dir: Directory to save LaTeX tables
    """
    logger.info("Generating LaTeX tables")
    
    latex_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    if 'model_name' not in results_df.columns:
        from utils import make_model_name
        results_df['model_name'] = results_df.apply(lambda row: make_model_name({
            'counterfactual_method': row['counterfactual_method'],
            'fairness_constraint': row['fairness_constraint'],
            'adjustment_strength': row['adjustment_strength'],
            'amplification_factor': row['amplification_factor']
        }), axis=1)

    for dataset in results_df['dataset'].unique():
        # Filter results for this dataset
        ds_results = results_df[results_df['dataset'] == dataset]
        
        try:
            # Find best model (by demographic parity improvement)
            best_model = ds_results.iloc[ds_results['improvement_demographic_parity_difference'].idxmax()]
            
            # Map results to benchmark format
            benchmark_format = {
                'BA': best_model['fair_performance_balanced_accuracy'],
                'Acc': best_model['fair_performance_accuracy'],
                'SP': best_model['fair_fairness_demographic_parity_difference'],
                'EO': best_model['fair_fairness_equal_opportunity_difference'],
                'EOd': best_model['fair_fairness_equalized_odds_difference'],
                'PP': best_model.get('fair_fairness_predictive_parity_difference', 0),
                'PE': best_model.get('fair_fairness_predictive_equality_difference', 0),
                'TE': best_model.get('fair_fairness_treatment_equality_difference', 0),
                'ABROCA': best_model.get('fair_fairness_abroca', 0)
            }
            
            # Get baseline results
            baseline_df = baseline_results[dataset]
            
            # Add TSCFM row
            tscfm_row = pd.DataFrame([{'Model': 'TSCFM'} | benchmark_format])
            comparison = pd.concat([baseline_df, tscfm_row], ignore_index=True)

            latex_table = comparison.to_latex(
                index=False,
                float_format="%.4f",
                caption=f"Performance comparison on {dataset.capitalize()} dataset",
                label=f"tab:results_{dataset}"
            )

            latex_path = os.path.join(latex_dir, f"{dataset}_results.tex")
            with open(latex_path, 'w') as f:
                f.write(latex_table)
            
            logger.info(f"Generated LaTeX table for {dataset}")
        except Exception as e:
            logger.error(f"Error generating LaTeX table for {dataset}: {e}")
    
    # Generate method comparison table
    try:
        # Group by method and calculate mean improvement
        method_results = results_df.groupby('counterfactual_method')[
            ['improvement_demographic_parity_difference', 
             'improvement_equal_opportunity_difference',
             'fair_performance_accuracy']
        ].mean().reset_index()
        
        # Create LaTeX table
        method_latex = method_results.to_latex(
            index=False,
            float_format="%.4f",
            caption="Comparison of counterfactual generation methods",
            label="tab:method_comparison"
        )
        
        method_path = os.path.join(latex_dir, "method_comparison.tex")
        with open(method_path, 'w') as f:
            f.write(method_latex)
        
        logger.info("Generated method comparison LaTeX table")
    except Exception as e:
        logger.error(f"Error generating method comparison table: {e}")
    
    logger.info("Completed LaTeX table generation")

def generate_fairness_comparison_tables(results_df, output_dir):
    """
    Generate comprehensive LaTeX tables comparing fairness constraints.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save tables
    """
    latex_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)
    
    # 1. Overall comparison table across all datasets and methods
    overall_df = results_df.groupby('fairness_constraint')[
        ['fair_performance_accuracy', 
         'improvement_demographic_parity_difference', 
         'improvement_equal_opportunity_difference']
    ].mean().reset_index()
    
    overall_df.columns = [
        'Fairness Constraint', 
        'Accuracy', 
        'DP Improvement (\%)', 
        'EO Improvement (\%)'
    ]
    
    overall_df['Fairness Constraint'] = overall_df['Fairness Constraint'].apply(
        lambda x: x.replace('_difference', '').replace('_', ' ').title()
    )

    overall_latex = overall_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Comparison of Fairness Constraints Across All Experiments",
        label="tab:fairness_constraints_comparison"
    )
    
    with open(os.path.join(latex_dir, "fairness_constraints_comparison.tex"), 'w') as f:
        f.write(overall_latex)
    
    # 2. Dataset-specific tables
    for dataset in results_df['dataset'].unique():
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        # Average across methods for each constraint
        dataset_summary = dataset_df.groupby('fairness_constraint')[
            ['fair_performance_accuracy', 
             'improvement_demographic_parity_difference', 
             'improvement_equal_opportunity_difference', 
             'improvement_equalized_odds_difference']
        ].mean().reset_index()

        dataset_summary.columns = [
            'Fairness Constraint', 
            'Accuracy', 
            'DP Improvement (\%)', 
            'EO Improvement (\%)',
            'EOd Improvement (\%)'
        ]

        dataset_summary['Fairness Constraint'] = dataset_summary['Fairness Constraint'].apply(
            lambda x: x.replace('_difference', '').replace('_', ' ').title()
        )

        dataset_latex = dataset_summary.to_latex(
            index=False,
            float_format="%.3f",
            caption=f"Fairness Constraints Comparison on {dataset.capitalize()} Dataset",
            label=f"tab:fairness_constraints_{dataset}"
        )

        with open(os.path.join(latex_dir, f"{dataset}_fairness_comparison.tex"), 'w') as f:
            f.write(dataset_latex)
    
    # 3. Detailed table for each dataset
    for dataset in results_df['dataset'].unique():
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        # Group by constraint and method
        detailed_df = dataset_df.groupby(['fairness_constraint', 'counterfactual_method'])[
            ['fair_performance_accuracy', 
             'fair_fairness_demographic_parity_difference',
             'fair_fairness_equal_opportunity_difference',
             'improvement_demographic_parity_difference', 
             'improvement_equal_opportunity_difference']
        ].mean().reset_index()

        detailed_df.columns = [
            'Fairness Constraint',
            'Counterfactual Method',
            'Accuracy',
            'DP Difference',
            'EO Difference',
            'DP Improvement (\%)',
            'EO Improvement (\%)'
        ]

        detailed_df['Fairness Constraint'] = detailed_df['Fairness Constraint'].apply(
            lambda x: x.replace('_difference', '').replace('_', ' ').title()
        )
        detailed_df['Counterfactual Method'] = detailed_df['Counterfactual Method'].apply(
            lambda x: x.title().replace('_', ' ')
        )

        detailed_latex = detailed_df.to_latex(
            index=False,
            float_format="%.3f",
            caption=f"Detailed Fairness Analysis on {dataset.capitalize()} Dataset",
            label=f"tab:detailed_fairness_{dataset}"
        )
        with open(os.path.join(latex_dir, f"{dataset}_detailed_fairness.tex"), 'w') as f:
            f.write(detailed_latex)
    
    # 4. Best results table
    best_results = []
    for dataset in results_df['dataset'].unique():
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        # Find best DP improvement
        best_dp_idx = dataset_df['improvement_demographic_parity_difference'].idxmax()
        best_dp = dataset_df.loc[best_dp_idx]
        
        # Find best accuracy
        best_acc_idx = dataset_df['fair_performance_accuracy'].idxmax()
        best_acc = dataset_df.loc[best_acc_idx]
        
        # Add to best results
        best_results.append({
            'Dataset': dataset.capitalize(),
            'Best DP Config': f"{best_dp['fairness_constraint'].replace('_difference', '').title()} + {best_dp['counterfactual_method'].title()}",
            'Best DP Improvement': best_dp['improvement_demographic_parity_difference'],
            'DP Config Accuracy': best_dp['fair_performance_accuracy'],
            'Best Acc Config': f"{best_acc['fairness_constraint'].replace('_difference', '').title()} + {best_acc['counterfactual_method'].title()}",
            'Best Accuracy': best_acc['fair_performance_accuracy'],
            'Acc Config DP Improvement': best_acc['improvement_demographic_parity_difference']
        })

    best_df = pd.DataFrame(best_results)

    best_latex = best_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Best Configurations by Dataset",
        label="tab:best_configurations"
    )
    
    with open(os.path.join(latex_dir, "best_configurations.tex"), 'w') as f:
        f.write(best_latex)
        
    logger.info(f"Generated LaTeX comparison tables in {latex_dir}")

def generate_fairness_visualizations(results_df, output_dir):
    """
    Generate all fairness comparison visualizations.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations", "fairness_analysis")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Overall fairness constraint comparison
    overall_path = os.path.join(viz_dir, "overall_constraint_comparison.png")
    try:
        plot_fairness_constraint_comparison(results_df, output_path=overall_path)
        logger.info(f"Generated overall fairness constraint comparison")
    except Exception as e:
        logger.error(f"Error generating overall fairness comparison: {e}")
    
    # 2. Dataset-specific comparisons
    for dataset in results_df['dataset'].unique():
        dataset_path = os.path.join(viz_dir, f"{dataset}_constraint_comparison.png")
        try:
            plot_fairness_constraint_comparison(results_df, dataset=dataset, output_path=dataset_path)
            logger.info(f"Generated {dataset} fairness constraint comparison")
        except Exception as e:
            logger.error(f"Error generating {dataset} fairness comparison: {e}")
    
    # 3. Method-specific comparisons
    for method in results_df['counterfactual_method'].unique():
        method_path = os.path.join(viz_dir, f"{method}_constraint_comparison.png")
        try:
            plot_fairness_constraint_comparison(results_df, method=method, output_path=method_path)
            logger.info(f"Generated {method} fairness constraint comparison")
        except Exception as e:
            logger.error(f"Error generating {method} fairness comparison: {e}")
    
    # 4. Trade-off scatter plots
    fairness_metrics = [
        'demographic_parity_difference',
        'equal_opportunity_difference',
        'equalized_odds_difference'
    ]
    
    for metric in fairness_metrics:
        try:
            metric_path = os.path.join(viz_dir, f"tradeoff_{metric.replace('_difference', '')}.png")
            plot_trade_off_scatter(results_df, fairness_metric=metric, output_path=metric_path)
            logger.info(f"Generated trade-off scatter plot for {metric}")
        except Exception as e:
            logger.error(f"Error generating trade-off plot for {metric}: {e}")
        
        # Dataset-specific trade-offs
        for dataset in results_df['dataset'].unique():
            try:
                dataset_metric_path = os.path.join(viz_dir, f"{dataset}_tradeoff_{metric.replace('_difference', '')}.png")
                plot_trade_off_scatter(results_df, dataset=dataset, fairness_metric=metric, output_path=dataset_metric_path)
                logger.info(f"Generated {dataset} trade-off scatter plot for {metric}")
            except Exception as e:
                logger.error(f"Error generating {dataset} trade-off plot for {metric}: {e}")


    
    logger.info(f"Generated fairness visualizations in {viz_dir}")

def main():
    """Main function."""
    args = parse_args()
    start_time = time.time()
    
    # Determine results directory
    if args.results_dir is None:
        results_dir = find_latest_results_dir()
        logger.info(f"Using latest results directory: {results_dir}")
    else:
        results_dir = args.results_dir
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(results_dir, "compiled")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    # Compile results
    results_df = compile_experiment_results(results_dir, output_dir, args.dataset)
    
    if results_df is None or results_df.empty:
        logger.error("No valid results found. Exiting.")
        return
    
    # Generate visualizations
    generate_combined_visualizations(results_df, output_dir, args.dataset)
    
    # Generate fairness comparison visualizations
    generate_fairness_visualizations(results_df, output_dir)
    
    # Generate LaTeX tables
    if args.generate_latex:
        from baseline_results import BENCHMARK_RESULTS
        generate_latex_tables(results_df, BENCHMARK_RESULTS, output_dir)
        
        # Generate fairness comparison tables
        generate_fairness_comparison_tables(results_df, output_dir)
    
    logger.info(f"Results compilation completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Output saved to {output_dir}")

if __name__ == "__main__":
    main()
