import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from typing import List, Dict, Tuple, Optional

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def load_results(file_path: str) -> pd.DataFrame:
    """
    Load experiment results from CSV file.
    
    Args:
        file_path: Path to the CSV file with results
        
    Returns:
        DataFrame with loaded results with model types identified
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} experiment results from {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Add a column to identify model type (baseline vs TSCFM)
    df['is_tscfm'] = df['Model'].apply(lambda x: bool(re.match(r'^[GSM]-', x)) if isinstance(x, str) else False)
    
    # First 22 models are from the study (baseline models)
    if len(df) > 22:
        df.loc[:21, 'is_tscfm'] = False
    
    return df

def create_fairness_focused_plots(results_df: pd.DataFrame, 
                                output_dir: str,
                                file_prefix: str = 'german') -> None:
    """
    Create three scatter plots, one for each fairness metric (SP, EO, EOd).
    Each plot highlights top TSCFM models by different criteria.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save output files
        file_prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify TSCFM models if not already done
    if 'is_tscfm' not in results_df.columns:
        results_df['is_tscfm'] = results_df['Model'].apply(
            lambda x: bool(re.match(r'^[GSM]-', x)) if isinstance(x, str) else False
        )
        
        # First 22 models are from the study (baseline models)
        if len(results_df) > 22:
            results_df.loc[:21, 'is_tscfm'] = False
    

    for metric in ['SP', 'EO', 'EOd']:
        results_df[f"{metric}_abs"] = results_df[metric].abs()

    fairness_metrics = [
        ('SP_abs', 'Demographic Parity', 'DP'),
        ('EO_abs', 'Equal Opportunity', 'EO'),
        ('EOd_abs', 'Equalized Odds', 'EOd')
    ]
    

    for fair_metric, fair_name, model_pattern in fairness_metrics:
        # Get all TSCFM models (regardless of optimization type)
        tscfm_df = results_df[results_df['is_tscfm']].copy()
        
        if len(tscfm_df) == 0:
            print("Warning: No TSCFM models found")
            continue
        
        # Find optimization type for each model
        tscfm_df['optimization_type'] = tscfm_df['Model'].apply(
            lambda x: re.search(r'-([^-]+)-', x).group(1) if re.search(r'-([^-]+)-', x) else None
        )
        
        # Filter to models optimizing for this specific fairness metric
        specific_models = tscfm_df[tscfm_df['optimization_type'] == model_pattern]
        
        if len(specific_models) == 0:
            print(f"Warning: No TSCFM models found optimizing for {fair_name}")
            continue
            
        metric_base = fair_metric.replace('_abs', '')
        
        # Get top models by different criteria - from the filtered models
        top_by_accuracy = specific_models.sort_values('Acc', ascending=False).head(5)['Model'].tolist()
        top_by_fairness = specific_models.sort_values(fair_metric, ascending=True).head(5)['Model'].tolist()
        
        # Calculate balanced score
        # Normalize accuracy (higher is better)
        acc_min, acc_max = specific_models['Acc'].min(), specific_models['Acc'].max()
        if acc_max > acc_min:
            specific_models['acc_normalized'] = (specific_models['Acc'] - acc_min) / (acc_max - acc_min)
        else:
            specific_models['acc_normalized'] = 1.0
        
        # Normalize fairness (lower absolute value is better)
        fair_min, fair_max = specific_models[fair_metric].min(), specific_models[fair_metric].max()
        if fair_max > fair_min:
            specific_models['fair_normalized'] = 1 - (specific_models[fair_metric] - fair_min) / (fair_max - fair_min)
        else:
            specific_models['fair_normalized'] = 1.0
        
        # Calculate harmonic mean
        numerator = 2 * specific_models['acc_normalized'] * specific_models['fair_normalized']
        denominator = specific_models['acc_normalized'] + specific_models['fair_normalized']
        denominator = denominator.replace(0, 1e-10)  # Avoid division by zero
        
        specific_models['balanced_score'] = numerator / denominator
        
        # Get top models by balanced score
        top_by_balance = specific_models.sort_values('balanced_score', ascending=False).head(5)['Model'].tolist()
        
        # For visual comparison, highlight all TSCFM models of this specific type
        specific_model_mask = tscfm_df['optimization_type'] == model_pattern
        other_tscfm_mask = ~specific_model_mask
        
        plot_fairness_tradeoff(
            results_df=results_df,
            specific_models=tscfm_df[specific_model_mask],
            other_models=tscfm_df[other_tscfm_mask],
            acc_metric='Acc',
            fair_metric=fair_metric,
            fair_name=fair_name,
            top_models={
                'Top Accuracy': top_by_accuracy,
                f'Top {fair_name}': top_by_fairness,
                f'Balanced {metric_base}': top_by_balance
            },
            output_path=os.path.join(output_dir, f"{file_prefix}_{metric_base}_optimization.png")
        )

def plot_fairness_tradeoff(results_df, specific_models, other_models, 
                         acc_metric, fair_metric, fair_name, 
                         top_models, output_path):
    """
    Create a scatter plot showing the trade-off between accuracy and fairness
    with better highlighting of model types.
    
    Args:
        results_df: DataFrame with all results
        specific_models: DataFrame with models optimized for this metric
        other_models: DataFrame with other TSCFM models
        acc_metric: Accuracy metric name
        fair_metric: Fairness metric name
        fair_name: Display name for the fairness metric
        top_models: Dictionary mapping objective names to lists of top model names
        output_path: Path to save the plot
    """
    try:
        # Import adjustText if available (better label positioning)
        from adjustText import adjust_text
        has_adjust_text = True
    except ImportError:
        has_adjust_text = False
        print("Warning: adjustText library not found. Install with 'pip install adjustText' for better label positioning.")

    colors = {
        'Top Accuracy': 'blue',
        f'Top {fair_name}': 'green',
        f'Balanced {fair_metric.replace("_abs", "")}': 'red'
    }
    

    plt.figure(figsize=(14, 12))  # Increase figure size for better label spacing

    baseline_mask = ~results_df['is_tscfm']
    
    # Plot baseline models
    plt.scatter(results_df[baseline_mask][acc_metric], results_df[baseline_mask][fair_metric], 
               alpha=0.4, marker='x', color='darkgray', s=40, label='Baseline Models')
    
    # Plot other TSCFM models (not optimizing for this metric)
    if len(other_models) > 0:
        plt.scatter(other_models[acc_metric], other_models[fair_metric], 
                   alpha=0.3, color='lightgray', s=40, label='Other TSCFM Models')
    
    # Plot all TSCFM models optimizing for this metric
    plt.scatter(specific_models[acc_metric], specific_models[fair_metric], 
               alpha=0.6, color='gray', s=60, label=f'Models optimizing for {fair_name}')

    texts = []

    legend_elements = []
    
    for objective, model_list in top_models.items():
        color = colors.get(objective, 'purple')
        
        model_mask = results_df['Model'].isin(model_list)
        if not any(model_mask):
            print(f"Warning: No models found for {objective}")
            continue

        plt.scatter(results_df[model_mask][acc_metric], results_df[model_mask][fair_metric], 
                   alpha=1.0, color=color, s=100, marker='o')
 
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'{objective} (Top 5)'))
 
        for _, row in results_df[model_mask].iterrows():
            t = plt.text(row[acc_metric], row[fair_metric], row['Model'],
                     fontsize=9, color=color, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=2))
            texts.append(t)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    plt.xlabel(f'Accuracy ({acc_metric})', fontsize=12)
    plt.ylabel(f'|{fair_metric.replace("_abs", "")}| - {fair_name} Difference', fontsize=12)
    plt.title(f'Accuracy vs. {fair_name} Trade-off', fontsize=14)
    
    plt.grid(True, alpha=0.3)

    base_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label=f'Models optimizing for {fair_name}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                  markersize=10, alpha=0.3, label='Other TSCFM Models'),
        plt.Line2D([0], [0], marker='x', color='darkgray', 
                  markersize=10, label='Baseline Models')
    ]
    
    plt.legend(handles=base_legend + legend_elements, loc='upper right')

    if has_adjust_text and texts:
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                   expand_points=(1.5, 1.5),
                   force_points=(0.1, 0.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trade-off plot to {output_path}")

def get_top_n_models(df: pd.DataFrame, objective_metrics: List[Tuple[str, bool]], n: int = 5) -> List[str]:
    """
    Get top N models based on the specified objective metrics.
    
    Args:
        df: DataFrame with the metrics
        objective_metrics: List of (metric, higher_is_better) tuples
        n: Number of top models to return
        
    Returns:
        List of model names for the top models
    """

    df_copy = df.copy()
    
    # Calculate a combined score for multi-metric objectives
    if len(objective_metrics) > 1 or objective_metrics[0][0] == 'balanced_score':
        # Single metric that's already a combined score
        if objective_metrics[0][0] == 'balanced_score':
            metric, higher_is_better = objective_metrics[0]
            sorting_order = False if higher_is_better else True  # False = descending, True = ascending
            sorted_df = df_copy.sort_values(metric, ascending=sorting_order)
            return sorted_df.head(n)['Model'].tolist()
        
        # Multiple metrics to combine
        for metric, higher_is_better in objective_metrics:
            if metric not in df_copy.columns:
                continue
                
            # Normalize metric to 0-1 range
            min_val, max_val = df_copy[metric].min(), df_copy[metric].max()
            if max_val > min_val:
                if higher_is_better:
                    df_copy[f"{metric}_norm"] = (df_copy[metric] - min_val) / (max_val - min_val)
                else:
                    df_copy[f"{metric}_norm"] = 1 - (df_copy[metric] - min_val) / (max_val - min_val)
            else:
                df_copy[f"{metric}_norm"] = 1.0  # All values are equal
        
        # Calculate combined score (average of normalized metrics)
        norm_cols = [f"{m}_norm" for m, _ in objective_metrics if f"{m}_norm" in df_copy.columns]
        if norm_cols:
            df_copy['combined_score'] = df_copy[norm_cols].mean(axis=1)
            sorted_df = df_copy.sort_values('combined_score', ascending=False)
            return sorted_df.head(n)['Model'].tolist()
    
    # Single metric case
    metric, higher_is_better = objective_metrics[0]
    sorting_order = not higher_is_better  # False = descending, True = ascending
    sorted_df = df_copy.sort_values(metric, ascending=sorting_order)
    return sorted_df.head(n)['Model'].tolist()

def compute_balanced_score(df: pd.DataFrame, acc_metric: str, fair_metric: str) -> List[Tuple[str, bool]]:
    """
    Compute a balanced score between accuracy and fairness.
    
    Args:
        df: DataFrame with the metrics
        acc_metric: Accuracy metric (higher is better)
        fair_metric: Fairness metric (lower is better)
        
    Returns:
        List of (metric, higher_is_better) tuples
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Normalize accuracy (higher is better)
    acc_min, acc_max = df[acc_metric].min(), df[acc_metric].max()
    if acc_max > acc_min:
        df['acc_normalized'] = (df[acc_metric] - acc_min) / (acc_max - acc_min)
    else:
        df['acc_normalized'] = 1.0
    
    # Normalize fairness (lower absolute value is better)
    fair_min, fair_max = df[fair_metric].min(), df[fair_metric].max()
    if fair_max > fair_min:
        df['fair_normalized'] = 1 - (df[fair_metric] - fair_min) / (fair_max - fair_min)
    else:
        df['fair_normalized'] = 1.0
    
    # Calculate harmonic mean (balanced score favors models that are good at both)
    numerator = 2 * df['acc_normalized'] * df['fair_normalized']
    denominator = df['acc_normalized'] + df['fair_normalized']
    denominator = denominator.replace(0, float('nan'))  # Avoid division by zero
    
    df['balanced_score'] = numerator / denominator
    df['balanced_score'] = df['balanced_score'].fillna(0)  # Replace NaNs with 0

    return [('balanced_score', True)]

def create_trade_off_plot(df: pd.DataFrame, 
                        acc_metric: str, 
                        fair_metric: str,
                        fair_name: str,
                        top_models: Dict[str, List[str]],
                        output_path: str) -> None:
    """
    Create a scatter plot showing the trade-off between accuracy and fairness
    with better label positioning.
    
    Args:
        df: DataFrame with the metrics
        acc_metric: Accuracy metric name
        fair_metric: Fairness metric name
        fair_name: Display name for the fairness metric
        top_models: Dictionary mapping objective names to lists of top model names
        output_path: Path to save the plot
    """
    try:
        # Import adjustText if available (better label positioning)
        from adjustText import adjust_text
        has_adjust_text = True
    except ImportError:
        has_adjust_text = False
        print("Warning: adjustText library not found. Install with 'pip install adjustText' for better label positioning.")

    colors = {
        'Top Accuracy': 'blue',
        f'Top {fair_name}': 'green',
        f'Balanced {fair_metric.replace("_abs", "")}': 'red'
    }
 
    plt.figure(figsize=(14, 12))  # Increase figure size for better label spacing
    
    # Check if 'is_tscfm' column exists, otherwise create it
    if 'is_tscfm' not in df.columns:
        is_tscfm = df['Model'].apply(lambda x: bool(re.match(r'^[GSM]-', x)) if isinstance(x, str) else False)
        
        # First 22 models are baseline if we have many models
        if len(df) > 22:
            is_tscfm.iloc[:22] = False
    else:
        is_tscfm = df['is_tscfm']
    
    # Plot all TSCFM models
    plt.scatter(df[is_tscfm][acc_metric], df[is_tscfm][fair_metric], 
               alpha=0.5, color='lightgray', s=50, label='All TSCFM Models')
    
    # Plot baseline models
    plt.scatter(df[~is_tscfm][acc_metric], df[~is_tscfm][fair_metric], 
               alpha=0.5, marker='x', color='darkgray', s=50, label='Baseline Models')
    
    # Plot top models with different colors
    legend_elements = []  # Keep track of legend elements
    
    # Store text objects for later adjustment
    texts = []
    
    for objective, model_list in top_models.items():
        # Get color for this objective
        color = colors.get(objective, 'purple')  # Default to purple if not found
        
        model_mask = df['Model'].isin(model_list)
        if not any(model_mask):
            print(f"Warning: No models found for {objective}")
            continue
            
        # Plot the models
        plt.scatter(df[model_mask][acc_metric], df[model_mask][fair_metric], 
                   alpha=0.8, color=color, s=100, marker='o')

        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'{objective} (Top 5)'))
        
        # Add labels for these models
        for _, row in df[model_mask].iterrows():
            t = plt.text(row[acc_metric], row[fair_metric], row['Model'],
                     fontsize=9, color=color, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=2))
            texts.append(t)
    
    # Add a reference line for perfect fairness
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel(f'Accuracy ({acc_metric})', fontsize=12)
    plt.ylabel(f'|{fair_metric.replace("_abs", "")}| - {fair_name} Difference', fontsize=12)
    plt.title(f'Accuracy vs. {fair_name} Trade-off', fontsize=14)

    plt.grid(True, alpha=0.3)
    
    # Add TSCFM and baseline to legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                  markersize=10, label='All TSCFM Models'),
        plt.Line2D([0], [0], marker='x', color='darkgray', 
                  markersize=10, label='Baseline Models')
    ] + legend_elements
    
    # Add legend
    plt.legend(handles=legend_elements, loc='upper right')
    
    # If adjustText is available, use it to prevent label overlaps
    if has_adjust_text and texts:
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                   expand_points=(1.5, 1.5),
                   force_points=(0.1, 0.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trade-off plot to {output_path}")

def place_non_overlapping_labels(annotations):
    """
    Place labels with a simple force-directed algorithm to avoid overlaps.
    
    Args:
        annotations: List of dictionaries with 'text', 'x', 'y', and 'color' keys
    """
    if not annotations:
        return
        
    # Sort annotations by y-coordinate (helps with initial placement)
    annotations.sort(key=lambda a: a['y'])
    
    # Define text box properties
    fontsize = 9
    padding = 5
    box_width = max(len(a['text']) for a in annotations) * fontsize * 0.6  # Estimate width
    box_height = fontsize * 1.5
    
    # Initial placement - offset from points
    for i, a in enumerate(annotations):
        a['text_x'] = a['x'] + padding
        a['text_y'] = a['y'] + padding
        
        # Create the text annotation
        a['annotation'] = plt.annotate(
            a['text'],
            (a['x'], a['y']),
            xytext=(a['text_x'], a['text_y']),
            fontsize=fontsize,
            color=a['color'],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=a['color'], alpha=0.8)
        )
        
        # Get the bounding box
        renderer = plt.gcf().canvas.get_renderer()
        a['bbox'] = a['annotation'].get_bbox_patch().get_extents()
        
    # Adjust positions to avoid overlaps
    max_iterations = 100
    for _ in range(max_iterations):
        moved = False
        
        for i, a in enumerate(annotations):
            for j, b in enumerate(annotations):
                if i == j:
                    continue
                    
                # Check for overlap
                if rectangles_overlap(a['bbox'], b['bbox']):
                    # Move boxes away from each other
                    dx = (a['text_x'] - b['text_x']) * 0.1
                    dy = (a['text_y'] - b['text_y']) * 0.1
                    
                    # If boxes are at the same position, add a small random offset
                    if dx == 0 and dy == 0:
                        dx = np.random.uniform(-5, 5)
                        dy = np.random.uniform(-5, 5)
                    
                    # Apply movement
                    a['text_x'] += dx
                    a['text_y'] += dy
                    b['text_x'] -= dx
                    b['text_y'] -= dy
                    
                    # Update annotations
                    a['annotation'].set_position((a['text_x'], a['text_y']))
                    b['annotation'].set_position((b['text_x'], b['text_y']))
                    
                    # Update bounding boxes
                    renderer = plt.gcf().canvas.get_renderer()
                    a['bbox'] = a['annotation'].get_bbox_patch().get_extents()
                    b['bbox'] = b['annotation'].get_bbox_patch().get_extents()
                    
                    moved = True
        
        if not moved:
            break

def rectangles_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.
    
    Args:
        rect1: First rectangle (x0, y0, x1, y1)
        rect2: Second rectangle (x0, y0, x1, y1)
        
    Returns:
        True if rectangles overlap, False otherwise
    """
    x0_1, y0_1, x1_1, y1_1 = rect1.bounds
    x0_2, y0_2, x1_2, y1_2 = rect2.bounds
    
    return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)

# Main function to be called
def create_performance_summary(results_df: pd.DataFrame, 
                             output_dir: str,
                             file_prefix: str = 'german') -> None:
    """
    Create performance summary analyses and visualizations.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save output files
        file_prefix: Prefix for output files (e.g., dataset name)
    """
    # Add identification of TSCFM vs baseline models if not already present
    if 'is_tscfm' not in results_df.columns:
        results_df['is_tscfm'] = results_df['Model'].apply(
            lambda x: bool(re.match(r'^[GSM]-', x)) if isinstance(x, str) else False
        )
        
        # First 22 models are from the study (baseline models)
        if len(results_df) > 22:
            results_df.loc[:21, 'is_tscfm'] = False
    
    # Create the fairness-focused trade-off plots
    #create_fairness_focused_plots(results_df, output_dir, file_prefix)
    create_fairness_focused_plots_with_tscfm_only(results_df, output_dir, file_prefix)

def plot_tscfm_only_fairness_tradeoff(results_df, specific_models, other_models, 
                         acc_metric, fair_metric, fair_name, 
                         top_models, output_path):
    """
    Create a scatter plot showing the trade-off between accuracy and fairness
    with TSCFM models only (no baselines) for closer examination.
    
    Args:
        results_df: DataFrame with all results
        specific_models: DataFrame with models optimized for this metric
        other_models: DataFrame with other TSCFM models
        acc_metric: Accuracy metric name
        fair_metric: Fairness metric name
        fair_name: Display name for the fairness metric
        top_models: Dictionary mapping objective names to lists of top model names
        output_path: Path to save the plot
    """
    try:
        # Import adjustText if available (better label positioning)
        from adjustText import adjust_text
        has_adjust_text = True
    except ImportError:
        has_adjust_text = False
        print("Warning: adjustText library not found. Install with 'pip install adjustText' for better label positioning.")
    
    # Setup color mapping
    colors = {
        'Top Accuracy': 'blue',
        f'Top {fair_name}': 'green',
        f'Balanced {fair_metric.replace("_abs", "")}': 'red'
    }
    
    plt.figure(figsize=(14, 12))  # Increase figure size for better label spacing
    
    # Plot other TSCFM models (not optimizing for this metric)
    if len(other_models) > 0:
        plt.scatter(other_models[acc_metric], other_models[fair_metric], 
                   alpha=0.3, color='lightgray', s=40, label='Other TSCFM Models')
    
    # Plot all TSCFM models optimizing for this metric
    plt.scatter(specific_models[acc_metric], specific_models[fair_metric], 
               alpha=0.6, color='gray', s=60, label=f'Models optimizing for {fair_name}')
    
    # Store text objects for later adjustment
    texts = []
    
    # Plot top models with different colors
    legend_elements = []
    
    for objective, model_list in top_models.items():
        # Get color for this objective
        color = colors.get(objective, 'purple')
        
        # Filter to only keep TSCFM models from top model lists
        tscfm_mask = results_df['is_tscfm'] == True
        filtered_model_list = [model for model in model_list 
                              if model in results_df[tscfm_mask]['Model'].values]
        
        if not filtered_model_list:
            print(f"Warning: No TSCFM models found for {objective}")
            continue
            
        # Plot the models
        model_mask = results_df['Model'].isin(filtered_model_list)
        plt.scatter(results_df[model_mask][acc_metric], results_df[model_mask][fair_metric], 
                   alpha=1.0, color=color, s=100, marker='o')
        

        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'{objective} (Top 5)'))

        for _, row in results_df[model_mask].iterrows():
            t = plt.text(row[acc_metric], row[fair_metric], row['Model'],
                     fontsize=9, color=color, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=2))
            texts.append(t)
    
    # Add a reference line for perfect fairness
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel(f'Accuracy ({acc_metric})', fontsize=12)
    plt.ylabel(f'|{fair_metric.replace("_abs", "")}| - {fair_name} Difference', fontsize=12)
    plt.title(f'Accuracy vs. {fair_name} Trade-off (TSCFM Models Only)', fontsize=14)

    plt.grid(True, alpha=0.3)
    
    base_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label=f'Models optimizing for {fair_name}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                  markersize=10, alpha=0.3, label='Other TSCFM Models')
    ]
  
    plt.legend(handles=base_legend + legend_elements, loc='upper right')

    if has_adjust_text and texts:
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                   expand_points=(1.5, 1.5),
                   force_points=(0.1, 0.2))
  
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved TSCFM-only trade-off plot to {output_path}")

# Function to update the main fairness visualization function to include TSCFM-only plots
def create_fairness_focused_plots_with_tscfm_only(results_df: pd.DataFrame, 
                                output_dir: str,
                                file_prefix: str = 'german') -> None:
    """
    Create three sets of scatter plots, one for each fairness metric (SP, EO, EOd).
    Each set includes both the original plot with baselines and a TSCFM-only plot.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save output files
        file_prefix: Prefix for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify TSCFM models if not already done
    if 'is_tscfm' not in results_df.columns:
        results_df['is_tscfm'] = results_df['Model'].apply(
            lambda x: bool(re.match(r'^[GSM]-', x)) if isinstance(x, str) else False
        )
        
        # First 22 models are from the study (baseline models)
        if len(results_df) > 22:
            results_df.loc[:21, 'is_tscfm'] = False
    
    # Add columns for absolute values of fairness metrics
    for metric in ['SP', 'EO', 'EOd']:
        results_df[f"{metric}_abs"] = results_df[metric].abs()
    
    # Define fairness metrics to use with their corresponding model pattern
    fairness_metrics = [
        ('SP_abs', 'Demographic Parity', 'DP'),
        ('EO_abs', 'Equal Opportunity', 'EO'),
        ('EOd_abs', 'Equalized Odds', 'EOd')
    ]
    
    # Create one plot for each fairness metric
    for fair_metric, fair_name, model_pattern in fairness_metrics:
        # Get all TSCFM models (regardless of optimization type)
        tscfm_df = results_df[results_df['is_tscfm']].copy()
        
        if len(tscfm_df) == 0:
            print("Warning: No TSCFM models found")
            continue
        
        # Find optimization type for each model
        tscfm_df['optimization_type'] = tscfm_df['Model'].apply(
            lambda x: re.search(r'-([^-]+)-', x).group(1) if re.search(r'-([^-]+)-', x) else None
        )
        
        # Filter to models optimizing for this specific fairness metric
        specific_models = tscfm_df[tscfm_df['optimization_type'] == model_pattern]
        
        if len(specific_models) == 0:
            print(f"Warning: No TSCFM models found optimizing for {fair_name}")
            continue
            
        metric_base = fair_metric.replace('_abs', '')
        
        # Get top models by different criteria - from the filtered models
        top_by_accuracy = specific_models.sort_values('Acc', ascending=False).head(5)['Model'].tolist()
        top_by_fairness = specific_models.sort_values(fair_metric, ascending=True).head(5)['Model'].tolist()
        
        # Calculate balanced score
        # Normalize accuracy (higher is better)
        acc_min, acc_max = specific_models['Acc'].min(), specific_models['Acc'].max()
        if acc_max > acc_min:
            specific_models['acc_normalized'] = (specific_models['Acc'] - acc_min) / (acc_max - acc_min)
        else:
            specific_models['acc_normalized'] = 1.0
        
        # Normalize fairness (lower absolute value is better)
        fair_min, fair_max = specific_models[fair_metric].min(), specific_models[fair_metric].max()
        if fair_max > fair_min:
            specific_models['fair_normalized'] = 1 - (specific_models[fair_metric] - fair_min) / (fair_max - fair_min)
        else:
            specific_models['fair_normalized'] = 1.0
        
        # Calculate harmonic mean
        numerator = 2 * specific_models['acc_normalized'] * specific_models['fair_normalized']
        denominator = specific_models['acc_normalized'] + specific_models['fair_normalized']
        denominator = denominator.replace(0, 1e-10)  # Avoid division by zero
        
        specific_models['balanced_score'] = numerator / denominator
        
        # Get top models by balanced score
        top_by_balance = specific_models.sort_values('balanced_score', ascending=False).head(5)['Model'].tolist()
        
        # For visual comparison, highlight all TSCFM models of this specific type
        specific_model_mask = tscfm_df['optimization_type'] == model_pattern
        other_tscfm_mask = ~specific_model_mask
        
        # Prepare top models dictionary
        top_models_dict = {
            'Top Accuracy': top_by_accuracy,
            f'Top {fair_name}': top_by_fairness,
            f'Balanced {metric_base}': top_by_balance
        }
        
        # 1. Create the original plot with baselines
        plot_fairness_tradeoff(
            results_df=results_df,
            specific_models=tscfm_df[specific_model_mask],
            other_models=tscfm_df[other_tscfm_mask],
            acc_metric='Acc',
            fair_metric=fair_metric,
            fair_name=fair_name,
            top_models=top_models_dict,
            output_path=os.path.join(output_dir, f"{file_prefix}_{metric_base}_optimization.png")
        )
        
        # 2. Create the TSCFM-only plot (no baselines)
        plot_tscfm_only_fairness_tradeoff(
            results_df=results_df,
            specific_models=tscfm_df[specific_model_mask],
            other_models=tscfm_df[other_tscfm_mask],
            acc_metric='Acc',
            fair_metric=fair_metric,
            fair_name=fair_name,
            top_models=top_models_dict,
            output_path=os.path.join(output_dir, f"{file_prefix}_{metric_base}_optimization_tscfm_only.png")
        )


# Example usage
if __name__ == "__main__":
    # Load results
    results_df = load_results(r"path_to_file")
    # Create performance summary
    create_performance_summary(results_df, "output_directory", "card")



