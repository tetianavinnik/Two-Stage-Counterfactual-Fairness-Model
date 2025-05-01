import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib
matplotlib.use('Agg')
from config import METRIC_DISPLAY_NAMES, PRIMARY_FAIRNESS_METRICS

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_performance_comparison(comparison_df, output_path=None, title=None):
    """
    Create bar chart comparing TSCFM performance with baseline models
    
    Args:
        comparison_df: DataFrame with model comparison data
        output_path: Path to save the plot (if None, just display)
        title: Plot title (if None, use default)
    
    Returns:
        Matplotlib figure
    """
    # Filter to relevant columns
    perf_metrics = ['Acc', 'BA']
    if all(metric in comparison_df.columns for metric in perf_metrics):
        df = comparison_df[['Model'] + perf_metrics].copy()
    else:
        raise ValueError("Comparison DataFrame must contain Model, Acc, and BA columns")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get TSCFM data and other models data
    tscfm_data = df[df['Model'] == 'TSCFM']
    other_models = df[df['Model'] != 'TSCFM'].sort_values('Acc', ascending=False).head(8)  # Top 8 models
    
    # Combine for plotting
    plot_df = pd.concat([tscfm_data, other_models])
  
    models = plot_df['Model'].values
    indices = np.arange(len(models))
    width = 0.35

    ax.bar(indices - width/2, plot_df['Acc'], width, label='Accuracy', color='#3498db')
    ax.bar(indices + width/2, plot_df['BA'], width, label='Balanced Accuracy', color='#2ecc71')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(title or 'Performance Comparison')
    ax.set_xticks(indices)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    tscfm_idx = np.where(models == 'TSCFM')[0][0]
    ax.get_children()[tscfm_idx].set_color('darkblue')
    ax.get_children()[tscfm_idx + len(models)].set_color('darkgreen')

    for i, v in enumerate(plot_df['Acc']):
        ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(plot_df['BA']):
        ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_fairness_performance_scatter(comparison_df, fairness_metric='SP', output_path=None, 
                                     show_only_matching_constraint=True):
    """
    Create scatter plot of fairness vs. performance
    
    Args:
        comparison_df: DataFrame with model comparison data
        fairness_metric: Fairness metric to plot (e.g., 'SP', 'EO')
        output_path: Path to save the plot (if None, just display)
        show_only_matching_constraint: If True, only show TSCFM models that optimize for this metric
    
    Returns:
        Matplotlib figure
    """
    try:
        from adjustText import adjust_text
    except ImportError:
        print("Warning: adjustText library not found. Install with 'pip install adjustText' for better label placement.")
        adjust_text = None
    
    from utils import make_model_name
    
    # Check if required columns exist
    if not all(col in comparison_df.columns for col in ['Model', 'Acc', fairness_metric]):
        raise ValueError(f"Comparison DataFrame must contain Model, Acc, and {fairness_metric} columns")

    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Prepare data - take absolute value of fairness metric
    df = comparison_df.copy()
    df[f'abs_{fairness_metric}'] = df[fairness_metric].abs()

    print(f"Total models in data: {len(df)}")
    print(f"Model types: {df['Model'].unique()}")
    
    # Split into TSCFM and baseline models - use a better detection method
    # Look for models that follow the S-XX, M-XX, G-XX pattern or contain "TSCFM"
    tscfm_pattern = r'(^[SMG]-\w+)|(TSCFM)'
    is_tscfm = df['Model'].str.contains(tscfm_pattern, regex=True)
    tscfm_models = df[is_tscfm].copy()
    baseline_models = df[~is_tscfm].copy()
    
    print(f"Found {len(tscfm_models)} TSCFM models")
    print(f"Found {len(baseline_models)} baseline models")

    if show_only_matching_constraint and not tscfm_models.empty:
        # Map metrics to constraints
        metric_to_constraint = {
            'SP': 'DP',  # Demographic Parity
            'EO': 'EO',  # Equal Opportunity
            'EOd': 'EOd'  # Equalized Odds
        }
        
        # Extract constraint from model names and filter
        if fairness_metric in metric_to_constraint:
            target_constraint = metric_to_constraint[fairness_metric]
            
            # Filter based on pattern in model name (assumes naming pattern like S-DP-0.7-2.0)
            matching_models = tscfm_models[tscfm_models['Model'].str.contains(f'-{target_constraint}-', regex=False)]
            
            print(f"Filtered to {len(matching_models)} TSCFM models optimizing for {fairness_metric}")
            tscfm_models = matching_models

    ax.scatter(
        baseline_models['Acc'], 
        baseline_models[f'abs_{fairness_metric}'],
        s=100, 
        c='blue',
        alpha=0.7,
        label='Baseline Models'
    )

    if not tscfm_models.empty:
        ax.scatter(
            tscfm_models['Acc'], 
            tscfm_models[f'abs_{fairness_metric}'],
            s=150, 
            c='red',
            alpha=0.7,
            label='TSCFM Variants',
            edgecolor='black'
        )

    texts = []

    for i, row in baseline_models.iterrows():
        text = ax.text(
            row['Acc'], 
            row[f'abs_{fairness_metric}'],
            row['Model'],
            fontsize=9,
            color='black',
            ha='center',
            va='bottom'
        )
        texts.append(text)

    for i, row in tscfm_models.iterrows():
        text = ax.text(
            row['Acc'], 
            row[f'abs_{fairness_metric}'],
            row['Model'],
            fontsize=9,
            fontweight='bold',
            color='red',
            ha='center',
            va='bottom'
        )
        texts.append(text)

    if adjust_text is not None and texts:
        adjust_text(
            texts, 
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            expand_points=(1.5, 1.5),
            force_points=(0.2, 0.2),
            force_text=(0.5, 0.5)
        )

    ax.set_xlabel('Accuracy')
    ax.set_ylabel(f'|{METRIC_DISPLAY_NAMES.get(fairness_metric, fairness_metric)}|')

    metric_full_name = {
        'SP': 'Demographic Parity',
        'EO': 'Equal Opportunity',
        'EOd': 'Equalized Odds'
    }.get(fairness_metric, fairness_metric)
    
    ax.set_title(f'Fairness vs. Performance Trade-off ({metric_full_name})')

    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)

    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax.legend(loc='best')
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_fairness_metrics_comparison(baseline_results, fair_results, output_path=None):
    """
    Create a bar chart comparing fairness metrics before and after adjustments
    
    Args:
        baseline_results: Dictionary with baseline fairness metrics
        fair_results: Dictionary with fair model fairness metrics
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Determine which metrics to plot
    metrics = [m for m in PRIMARY_FAIRNESS_METRICS if m in baseline_results and m in fair_results]
    
    if not metrics:
        raise ValueError("No common fairness metrics found in both result sets")

    fig, ax = plt.subplots(figsize=(10, 6))

    indices = np.arange(len(metrics))
    width = 0.35

    baseline_values = [baseline_results[m] for m in metrics]
    fair_values = [fair_results[m] for m in metrics]

    ax.bar(indices - width/2, [abs(v) for v in baseline_values], width, label='Baseline Model', color='indianred')
    ax.bar(indices + width/2, [abs(v) for v in fair_values], width, label='Fair Model', color='seagreen')

    ax.set_xlabel('Fairness Metric')
    ax.set_ylabel('Absolute Value')
    ax.set_title('Fairness Metrics Before and After Adjustment')
    ax.set_xticks(indices)
    ax.set_xticklabels([METRIC_DISPLAY_NAMES.get(m, m) for m in metrics])
    ax.legend()

    for i, v in enumerate(baseline_values):
        ax.text(i - width/2, abs(v) + 0.005, f'{abs(v):.3f}', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(fair_values):
        ax.text(i + width/2, abs(v) + 0.005, f'{abs(v):.3f}', ha='center', va='bottom', fontsize=9)

    for i, (base, fair) in enumerate(zip(baseline_values, fair_values)):
        if abs(base) > 0:
            improvement = (abs(base) - abs(fair)) / abs(base) * 100
            ax.text(i, 0.002, f'↓ {improvement:.1f}%', ha='center', va='bottom', 
                    color='green' if improvement > 0 else 'red', fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_pareto_frontier(comparison_df, fairness_metric='SP', output_path=None):
    """
    Plot Pareto frontier of accuracy vs. fairness
    
    Args:
        comparison_df: DataFrame with model comparison data
        fairness_metric: Fairness metric to use
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Check if required columns exist
    if not all(col in comparison_df.columns for col in ['Model', 'Acc', fairness_metric]):
        raise ValueError(f"Comparison DataFrame must contain Model, Acc, and {fairness_metric} columns")

    fig, ax = plt.subplots(figsize=(10, 8))

    df = comparison_df.copy()
    df[f'abs_{fairness_metric}'] = df[fairness_metric].abs()
    
    # Find Pareto-optimal points
    pareto_models = []
    for idx, row in df.iterrows():
        is_dominated = False
        for idx2, row2 in df.iterrows():
            if idx != idx2:
                # Model 2 dominates Model 1 if it has better accuracy and fairness
                if (row2['Acc'] >= row['Acc'] and 
                    row2[f'abs_{fairness_metric}'] <= row[f'abs_{fairness_metric}'] and
                    (row2['Acc'] > row['Acc'] or 
                     row2[f'abs_{fairness_metric}'] < row[f'abs_{fairness_metric}'])):
                    is_dominated = False
                    break
        
        # If model is not dominated, add to Pareto frontier
        if not is_dominated:
            pareto_models.append(row)
    
    pareto_df = pd.DataFrame(pareto_models)

    ax.scatter(
        df['Acc'], 
        df[f'abs_{fairness_metric}'],
        s=80, 
        alpha=0.5,
        label='All Models'
    )

    pareto_df = pareto_df.sort_values('Acc')
    ax.plot(
        pareto_df['Acc'], 
        pareto_df[f'abs_{fairness_metric}'],
        'r-o',
        linewidth=2,
        markersize=10,
        label='Pareto Frontier'
    )

    tscfm_row = df[df['Model'] == 'TSCFM']
    if not tscfm_row.empty:
        tscfm_x = tscfm_row['Acc'].values[0]
        tscfm_y = tscfm_row[f'abs_{fairness_metric}'].values[0]
        ax.scatter(tscfm_x, tscfm_y, s=200, c='red', marker='*', label='TSCFM')

        ax.annotate(
            'TSCFM',
            (tscfm_x, tscfm_y),
            xytext=(10, -10),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )

    for i, model in enumerate(pareto_df['Model']):
        if model != 'TSCFM':  # TSCFM already labeled
            x = pareto_df['Acc'].iloc[i]
            y = pareto_df[f'abs_{fairness_metric}'].iloc[i]
            ax.annotate(
                model,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )

    ax.set_xlabel('Accuracy')
    ax.set_ylabel(f'|{METRIC_DISPLAY_NAMES.get(fairness_metric, fairness_metric)}|')
    ax.set_title(f'Pareto Frontier: Accuracy vs. {fairness_metric}')

    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_hyperparameter_sensitivity(results_df, param_name, metric_name, output_path=None, 
                                   use_boxplot=True, separate_by_constraint=True):
    """
    Plot sensitivity of a metric to a hyperparameter
    
    Args:
        results_df: DataFrame with experiment results
        param_name: Name of the hyperparameter column
        metric_name: Name of the metric column
        output_path: Path to save the plot (if None, just display)
        use_boxplot: Whether to use boxplots instead of line plots
        separate_by_constraint: Whether to separate by fairness constraint
    
    Returns:
        Matplotlib figure
    """
    # Check if required columns exist
    if not all(col in results_df.columns for col in [param_name, metric_name]):
        raise ValueError(f"Results DataFrame must contain {param_name} and {metric_name} columns")

    fig, ax = plt.subplots(figsize=(12, 8))
    
    if separate_by_constraint and 'fairness_constraint' in results_df.columns:
        # Get unique fairness constraints
        constraints = results_df['fairness_constraint'].unique()

        colors = {
            'demographic_parity_difference': 'blue',
            'equal_opportunity_difference': 'red',
            'equalized_odds_difference': 'green'
        }
        
        if use_boxplot:
            # Prepare data for boxplot
            boxplot_data = []
            positions = []
            labels = []
            colors_list = []
            
            # For each param value and constraint
            param_values = sorted(results_df[param_name].unique())
            width = 0.8 / len(constraints)
            
            for i, param_val in enumerate(param_values):
                for j, constraint in enumerate(constraints):
                    # Filter data for this param value and constraint
                    data = results_df[(results_df[param_name] == param_val) & 
                                    (results_df['fairness_constraint'] == constraint)][metric_name]
                    
                    if not data.empty:
                        boxplot_data.append(data)
                        pos = i + (j - len(constraints)/2 + 0.5) * width
                        positions.append(pos)

                        if j == 0:  # Only add the param value once
                            labels.append(param_val)
                        else:
                            labels.append('')
                        
                        colors_list.append(colors.get(constraint, 'gray'))

            boxplots = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, 
                               widths=width*0.9, showfliers=False)

            for box, color in zip(boxplots['boxes'], colors_list):
                box.set(facecolor=color, alpha=0.6)

            ax.set_xticks([i for i in range(len(param_values))])
            ax.set_xticklabels(param_values)

            legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=color, alpha=0.6, 
                                           label=constraint.replace('_difference', '').replace('_', ' ').title())
                             for constraint, color in colors.items() if constraint in constraints]
            ax.legend(handles=legend_elements, loc='best')
        
        else:
            for constraint in constraints:
                # Filter data for this constraint
                constraint_df = results_df[results_df['fairness_constraint'] == constraint]
                
                # Group by parameter value
                grouped = constraint_df.groupby(param_name)[metric_name].agg(['median', 'std']).reset_index()

                color = colors.get(constraint, 'gray')
                ax.plot(
                    grouped[param_name],
                    grouped['median'],
                    marker='o',
                    linestyle='-',
                    label=constraint.replace('_difference', '').replace('_', ' ').title(),
                    color=color
                )

                ax.fill_between(
                    grouped[param_name],
                    grouped['median'] - grouped['std'],
                    grouped['median'] + grouped['std'],
                    alpha=0.2,
                    color=color
                )

            ax.legend()
    
    else:  # Original behavior (no separation by constraint)
        if use_boxplot:
            # Create boxplot grouped by parameter value
            param_values = sorted(results_df[param_name].unique())
            
            # Collect data for each param value
            boxplot_data = [results_df[results_df[param_name] == val][metric_name] for val in param_values]

            ax.boxplot(boxplot_data, labels=param_values, patch_artist=True, 
                     widths=0.6, showfliers=False)

            for patch in ax.artists:
                patch.set_facecolor('teal')
                patch.set_alpha(0.6)
        
        else:  # Original behavior with mean/error bars
            # Group by parameter value and calculate median and std
            grouped = results_df.groupby(param_name)[metric_name].agg(['median', 'std']).reset_index()
            
            # Plot line with error bars
            ax.errorbar(
                grouped[param_name],
                grouped['median'],
                yerr=grouped['std'],
                marker='o',
                linestyle='-',
                capsize=5,
                markersize=8
            )

            for i, row in grouped.iterrows():
                ax.annotate(
                    f'{row["median"]:.3f}',
                    (row[param_name], row["median"]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center'
                )

    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'Sensitivity of {metric_name.replace("_", " ").title()} to {param_name.replace("_", " ").title()}')

    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_counterfactual_method_comparison(results_df, output_path=None):
    """
    Compare performance of different counterfactual methods
    
    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Check if required columns exist
    if 'counterfactual_method' not in results_df.columns:
        raise ValueError("Results DataFrame must contain 'counterfactual_method' column")
    
    # Add model name to the DataFrame if not already present
    if 'model_name' not in results_df.columns:
        from utils import make_model_name
        results_df['model_name'] = results_df.apply(lambda row: make_model_name({
            'counterfactual_method': row['counterfactual_method'],
            'fairness_constraint': row['fairness_constraint'],
            'adjustment_strength': row['adjustment_strength'],
            'amplification_factor': row['amplification_factor']
        }), axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    possible_perf_metrics = [
        ['fair_performance_accuracy', 'fair_performance_roc_auc'],
        ['fair_performance_accuracy', 'fair_performance_f1_score'],
        ['accuracy', 'roc_auc']
    ]
    
    perf_metrics = None
    for metrics in possible_perf_metrics:
        if all(col in results_df.columns for col in metrics):
            perf_metrics = metrics
            break
    
    if perf_metrics is None:
        # No usable performance metrics found
        raise ValueError("No valid performance metrics found in DataFrame")
    

    fairness_patterns = [
        'fair_fairness_demographic_parity_difference',
        'fair_fairness_equal_opportunity_difference',
        'fair_fairness_equalized_odds_difference',
        'demographic_parity_difference',
        'equal_opportunity_difference',
        'equalized_odds_difference'
    ]
    
    available_fairness = [col for col in fairness_patterns if col in results_df.columns]
    
    # Group by method and calculate mean for performance metrics
    perf_by_method = results_df.groupby('counterfactual_method')[perf_metrics].mean().reset_index()
    
    methods = perf_by_method['counterfactual_method'].values
    indices = np.arange(len(methods))
    width = 0.35

    ax1.bar(indices - width/2, perf_by_method[perf_metrics[0]], width, 
            label=perf_metrics[0].split('_')[-1].capitalize(), color='#3498db')
    ax1.bar(indices + width/2, perf_by_method[perf_metrics[1]], width, 
            label=perf_metrics[1].split('_')[-1].capitalize(), color='#2ecc71')

    ax1.set_xlabel('Counterfactual Method')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance by Counterfactual Method')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(methods)
    ax1.legend()

    for i, v in enumerate(perf_by_method[perf_metrics[0]]):
        ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(perf_by_method[perf_metrics[1]]):
        ax1.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    if available_fairness:
        # Group by method and calculate mean absolute value for fairness metrics
        fairness_means = results_df.groupby('counterfactual_method')[available_fairness].mean().abs().reset_index()

        bar_width = 0.2
        offsets = np.linspace(-0.3, 0.3, len(available_fairness))

        for i, metric in enumerate(available_fairness):
            # Get display name
            metric_parts = metric.split('_')
            metric_display = '_'.join(metric_parts[-3:]) if len(metric_parts) >= 3 else metric
            
            ax2.bar(indices + offsets[i], fairness_means[metric], bar_width, 
                    label=metric_display)

        ax2.set_xlabel('Counterfactual Method')
        ax2.set_ylabel('Fairness Metric (absolute)')
        ax2.set_title('Fairness by Counterfactual Method')
        ax2.set_xticks(indices)
        ax2.set_xticklabels(methods)
        ax2.legend()

        max_val = fairness_means[available_fairness].max().max()
        ax2.set_ylim(0, min(0.2, max_val * 1.5))
    else:
        # If no fairness metrics available, check for improvement metrics
        improvement_patterns = [
            'improvement_demographic_parity_difference',
            'improvement_equal_opportunity_difference',
            'improvement_equalized_odds_difference'
        ]
        
        available_improvements = [col for col in improvement_patterns if col in results_df.columns]
        
        if available_improvements:
            # Group by method and calculate mean for improvement metrics
            improvement_means = results_df.groupby('counterfactual_method')[available_improvements].mean().reset_index()

            bar_width = 0.2
            offsets = np.linspace(-0.3, 0.3, len(available_improvements))

            for i, metric in enumerate(available_improvements):
                metric_parts = metric.split('_')
                metric_display = '_'.join(metric_parts[-3:]) if len(metric_parts) >= 3 else metric
                
                ax2.bar(indices + offsets[i], improvement_means[metric], bar_width, 
                        label=metric_display)

            ax2.set_xlabel('Counterfactual Method')
            ax2.set_ylabel('Fairness Improvement (%)')
            ax2.set_title('Fairness Improvement by Method')
            ax2.set_xticks(indices)
            ax2.set_xticklabels(methods)
            ax2.legend()
        else:
            # If no fairness or improvement metrics available, display a message
            ax2.text(0.5, 0.5, "No fairness metrics available", 
                     ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_fairness_constraint_comparison(results_df, dataset=None, method=None, output_path=None):
    """
    Create comparison plot showing how different fairness constraints perform.
    
    Args:
        results_df: DataFrame with experiment results
        dataset: Dataset to filter for (if None, average across datasets)
        method: Method to filter for (if None, average across methods)
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    df = results_df.copy()
    if dataset:
        df = df[df['dataset'] == dataset]
    if method:
        df = df[df['counterfactual_method'] == method]
    
    if len(df) == 0:
        raise ValueError("No data available after filtering")

    grouped = df.groupby('fairness_constraint')

    metrics = []
    for constraint, group in grouped:
        constraint_name = constraint.replace('_difference', '').replace('_', ' ').title()
        metrics.append({
            'Constraint': constraint_name,
            'Accuracy': group['fair_performance_accuracy'].mean(),
            'DP Improvement': group['improvement_demographic_parity_difference'].mean(),
            'EO Improvement': group['improvement_equal_opportunity_difference'].mean(),
            'EOd Improvement': group['improvement_equalized_odds_difference'].mean() if 'improvement_equalized_odds_difference' in group.columns else float('nan')
        })

    metrics_df = pd.DataFrame(metrics)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    constraints = metrics_df['Constraint']
    accuracy = metrics_df['Accuracy']
    ax1.bar(constraints, accuracy, color='teal')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Fairness Constraint')

    for i, v in enumerate(accuracy):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')

    x = np.arange(len(constraints))
    width = 0.25
    
    ax2.bar(x - width, metrics_df['DP Improvement'], width, 
           label='Demographic Parity', color='royalblue')
    ax2.bar(x, metrics_df['EO Improvement'], width, 
           label='Equal Opportunity', color='darkorange')
    ax2.bar(x + width, metrics_df['EOd Improvement'], width, 
           label='Equalized Odds', color='forestgreen')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(constraints)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Fairness Improvements by Constraint')
    ax2.legend()

    for i, metric in enumerate(['DP Improvement', 'EO Improvement', 'EOd Improvement']):
        for j, v in enumerate(metrics_df[metric]):
            if not np.isnan(v):
                ax2.text(j + (i-1)*width, v + 2, f'{v:.1f}%', ha='center')

    title = "Fairness Constraint Comparison"
    if dataset and method:
        title += f" - {dataset.capitalize()} with {method.capitalize()}"
    elif dataset:
        title += f" - {dataset.capitalize()}"
    elif method:
        title += f" - {method.capitalize()} Method"
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_trade_off_scatter(results_df, dataset=None, fairness_metric='demographic_parity_difference', 
                          output_path=None):
    """
    Create a scatter plot showing accuracy vs. fairness trade-off across different constraints.
    
    Args:
        results_df: DataFrame with experiment results
        dataset: Dataset to filter for (if None, use all datasets)
        fairness_metric: Fairness metric to use for comparison
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Import necessary libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    # Try to import model name utility
    try:
        from utils import make_model_name
        has_make_model_name = True
    except ImportError:
        has_make_model_name = False
    
    # Filter data if needed
    df = results_df.copy()
    if dataset:
        df = df[df['dataset'] == dataset]
    
    if len(df) == 0:
        raise ValueError("No data available after filtering")
    
    # Check for required columns
    improvement_col = f'improvement_{fairness_metric}'
    required_columns = ['fairness_constraint', 'counterfactual_method', 
                       'adjustment_strength', 'amplification_factor', 
                       'fair_performance_accuracy', improvement_col]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
 
    print(f"Total models in dataset: {len(df)}")

    constraint_method_counts = df.groupby(['fairness_constraint', 'counterfactual_method']).size()
    print("Models by constraint and method:")
    for (constraint, method), count in constraint_method_counts.items():
        print(f"  {constraint} + {method}: {count}")

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = {
        'demographic_parity_difference': '#1f77b4',  # Blue
        'equal_opportunity_difference': '#ff7f0e',   # Orange
        'equalized_odds_difference': '#2ca02c'       # Green
    }

    markers = {
        'structural_equation': 'o',  # Circle
        'matching': '^',            # Triangle
        'generative': 's'           # Square
    }
    
    marker_size = 100

    legend_entries = {}

    method_map = {
        'structural_equation': 'S',
        'matching': 'M',
        'generative': 'G'
    }
    
    constraint_map = {
        'demographic_parity_difference': 'DP',
        'equal_opportunity_difference': 'EO',
        'equalized_odds_difference': 'EOd'
    }

    plotted_models = 0

    position_map = defaultdict(list)

    for _, model in df.iterrows():
        constraint = model['fairness_constraint']
        method = model['counterfactual_method']
 
        color = colors.get(constraint, '#999999')  # Default gray for unknown constraints
        marker = markers.get(method, 'x')  # Default X for unknown methods

        x_pos = model['fair_performance_accuracy']
        y_pos = model[improvement_col]

        position_key = (round(x_pos, 6), round(y_pos, 6))

        position_map[position_key].append({
            'constraint': constraint,
            'method': method,
            'adjustment_strength': model['adjustment_strength'],
            'amplification_factor': model['amplification_factor'],
            'color': color
        })

        ax.scatter(
            x_pos,
            y_pos,
            color=color,
            marker=marker,
            s=marker_size,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5,
            zorder=10
        )
        
        plotted_models += 1

        constraint_name = constraint.replace('_difference', '').replace('_', ' ').title()
        method_name = method.replace('_', ' ').title()
        label = f"{constraint_name} + {method_name}"
        
        if label not in legend_entries:
            legend_entries[label] = plt.Line2D(
                [0], [0],
                marker=marker,
                color='w',
                markerfacecolor=color,
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=0.5,
                label=label
            )
    
    print(f"Actually plotted {plotted_models} models")
    
    # Find overlapping points
    overlapping_positions = {pos: models for pos, models in position_map.items() if len(models) > 1}
    if overlapping_positions:
        print(f"Found {len(overlapping_positions)} positions with overlapping points:")
        for pos, models in overlapping_positions.items():
            model_names = []
            for m in models:
                constraint_code = constraint_map.get(m['constraint'], m['constraint'].split('_')[0].upper())
                method_code = method_map.get(m['method'], m['method'][0].upper())
                model_names.append(f"{method_code}-{constraint_code}")
            print(f"  Position {pos}: {', '.join(model_names)}")
    
    # Second pass: Add labels with staggered positioning for overlaps
    for position, models in position_map.items():
        x_pos, y_pos = position

        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        base_x_offset = x_range * 0.005  # 0.5% of x-axis range
        base_y_offset = y_range * 0.02   # 2% of y-axis range

        x_mid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        if x_pos < x_mid:  # Left side of plot
            direction = 1  # Right
            ha = 'left'
        else:  # Right side of plot
            direction = -1  # Left
            ha = 'right'

        for i, model in enumerate(models):
            # Generate model name
            method_prefix = method_map.get(model['method'], model['method'][0].upper())
            constraint_code = constraint_map.get(model['constraint'], model['constraint'].split('_')[0].upper())

            adj_str = f"{model['adjustment_strength']:.1f}"
            amp_factor = f"{model['amplification_factor']:.1f}"
            
            # Standard format: S-EO-0.7-2.0
            model_name = f"{method_prefix}-{constraint_code}-{adj_str}-{amp_factor}"

            if has_make_model_name:
                try:
                    params = {
                        'counterfactual_method': model['method'],
                        'fairness_constraint': model['constraint'],
                        'adjustment_strength': model['adjustment_strength'],
                        'amplification_factor': model['amplification_factor']
                    }
                    model_name = make_model_name(params)
                except Exception:
                    # Keep using our format if make_model_name fails
                    pass

            if len(models) > 1:
                # Alternate between above and below the point
                if i % 2 == 0:
                    # Even indices go above
                    y_offset = base_y_offset * (1 + i//2)
                    extra_x_offset = 0 
                else:
                    # Odd indices go below
                    y_offset = -base_y_offset * (1 + i//2)
                    extra_x_offset = base_x_offset * direction * 0.5  # Slight horizontal offset
            else:
                y_offset = base_y_offset
                extra_x_offset = 0

            label_x = x_pos + (base_x_offset + extra_x_offset) * direction
            label_y = y_pos + y_offset

            ax.annotate(
                model_name,
                (x_pos, y_pos),
                xytext=(label_x, label_y),
                fontsize=9,
                color=model['color'],
                fontweight='bold',
                ha=ha,
                va='center',
                bbox=dict(
                    boxstyle="round,pad=0.2", 
                    facecolor='white', 
                    alpha=0.8,
                    edgecolor='none'
                ),
                zorder=20
            )
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Clean up the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    metric_name = fairness_metric.replace('_difference', '').replace('_', ' ').title()
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_name} Improvement (%)', fontsize=12, fontweight='bold')
    
    title = f"Accuracy vs. {metric_name} Improvement Trade-off"
    if dataset:
        title += f" - {dataset.capitalize()}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    handles = list(legend_entries.values())
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=10, frameon=True, framealpha=0.9)

    ax.grid(True, alpha=0.15, linestyle='-')

    ax.set_ylim(-2, 105)  # Allow a little margin below 0 and above 100

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)  # Make space for legend

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig
