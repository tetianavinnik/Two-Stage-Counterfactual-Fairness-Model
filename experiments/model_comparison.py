import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.colors as mcolors

def visualize_top_models_comparison(results_df: pd.DataFrame,
                                  output_dir: str,
                                  file_prefix: str = 'german',
                                  tscfm_base_acc: float = 0.75,
                                  tscfm_base_fairness: Dict[str, float] = None) -> None:
    """
    Create visualizations comparing top TSCFM models against baseline fair models.
    Shows changes in fairness and accuracy from the original base models.
    Focus on models with lowest fairness metric values.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save output files
        file_prefix: Prefix for output files
        tscfm_base_acc: Base accuracy for TSCFM models
        tscfm_base_fairness: Dictionary with base fairness values for TSCFM models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default fairness values if not provided
    if tscfm_base_fairness is None:
        tscfm_base_fairness = {'SP': 0.1, 'EO': 0.12, 'EOd': 0.15}
    
    # Identify TSCFM models by prefix
    results_df['is_tscfm'] = results_df['Model'].apply(
        lambda x: isinstance(x, str) and x.startswith(('G-', 'S-', 'M-'))
    )
    
    # Get TSCFM models
    tscfm_df = results_df[results_df['is_tscfm']].copy()
    
    if len(tscfm_df) == 0:
        print("Warning: No TSCFM models found")
        return
    
    # Extract counterfactual method and fairness type from model name
    tscfm_df['cf_method'] = tscfm_df['Model'].apply(
        lambda x: x[0] if isinstance(x, str) and len(x) > 0 else None
    )
    
    tscfm_df['fairness_type'] = tscfm_df['Model'].apply(
        lambda x: re.search(r'-([^-]+)-', x).group(1) if isinstance(x, str) and re.search(r'-([^-]+)-', x) else None
    )
    
    # Map method codes to full names
    method_names = {
        'G': 'Generative',
        'S': 'Structural Equation',
        'M': 'Matching'
    }
    
    # Create a mapping from method code to color
    method_colors = {
        'G': 'blue',
        'S': 'green',
        'M': 'red'
    }
    
    # Map fairness type codes to full names and corresponding metrics
    fairness_info = {
        'DP': {'name': 'Demographic Parity', 'metric': 'SP_abs', 'base_metric': 'SP'},
        'EO': {'name': 'Equal Opportunity', 'metric': 'EO_abs', 'base_metric': 'EO'},
        'EOd': {'name': 'Equalized Odds', 'metric': 'EOd_abs', 'base_metric': 'EOd'}
    }
    
    # Add absolute value columns if not present
    for base_metric in ['SP', 'EO', 'EOd']:
        abs_metric = f"{base_metric}_abs"
        if abs_metric not in results_df.columns:
            results_df[abs_metric] = results_df[base_metric].abs()
        if abs_metric not in tscfm_df.columns:
            tscfm_df[abs_metric] = tscfm_df[base_metric].abs()
    
    # Identify baseline models (those without fair prefixes)
    baseline_models = []
    baseline_fair_models = []
    
    # Regular expressions to identify baseline models and their fair versions
    baseline_pattern = r'^(kNN|DT|MLP|NB)$'
    fair_pattern = r'^(DIR|LFR|EOP|CEP)-([a-zA-Z0-9]+)$'
    
    # Compile patterns for faster matching
    baseline_regex = re.compile(baseline_pattern)
    fair_regex = re.compile(fair_pattern)
    
    # Dictionary to store base model metrics
    base_model_metrics = {}
    fair_model_metrics = {}
    
    # Process models to identify baselines and fair versions
    for _, row in results_df.iterrows():
        model_name = row['Model']
        
        if not isinstance(model_name, str):
            continue
            
        # Check if it's a baseline model
        if baseline_regex.match(model_name):
            baseline_models.append(model_name)
            
            # Store metrics for this baseline model
            base_model_metrics[model_name] = {
                'Acc': row['Acc']
            }
            
            # Store fairness metrics
            for metric in ['SP', 'EO', 'EOd']:
                if metric in row:
                    base_model_metrics[model_name][metric] = row[metric]
        
        # Check if it's a fair version of a baseline
        fair_match = fair_regex.match(model_name)
        if fair_match:
            base_name = fair_match.group(2)
            fair_model_metrics[model_name] = {
                'base_model': base_name,
                'Acc': row['Acc']
            }
            
            # Store fairness metrics
            for metric in ['SP', 'EO', 'EOd']:
                if metric in row:
                    fair_model_metrics[model_name][metric] = row[metric]
            
            # Add to list of fair models
            baseline_fair_models.append(model_name)
    
    # Create comparison data structure
    comparison_data = []
    
    # Process each fairness metric
    for fairness_code, fairness_details in fairness_info.items():
        fairness_name = fairness_details['name']
        fairness_metric = fairness_details['metric']
        base_fairness_metric = fairness_details['base_metric']
        
        # Find TSCFM models optimizing for this fairness metric
        fairness_models = tscfm_df[tscfm_df['fairness_type'] == fairness_code]
        
        if len(fairness_models) == 0:
            print(f"Warning: No TSCFM models found optimizing for {fairness_name}")
            continue
        
        # Find top model for each counterfactual method (lowest fairness metric value)
        for cf_method, method_name in method_names.items():
            method_models = fairness_models[fairness_models['cf_method'] == cf_method]
            
            if len(method_models) == 0:
                print(f"Warning: No {method_name} models found optimizing for {fairness_name}")
                continue
            
            # Get the model with lowest fairness metric value
            # Sort by fairness metric (ascending) to get model with lowest bias
            method_models = method_models.sort_values(fairness_metric)
            
            # Take top model with lowest fairness metric value
            if len(method_models) > 0:
                top_model = method_models.iloc[0]
                
                # Use the provided base accuracy and fairness for TSCFM
                base_acc = tscfm_base_acc
                base_fair = tscfm_base_fairness[base_fairness_metric]
                
                # Calculate changes with correct percentage formula
                acc_change = (top_model['Acc'] - base_acc) / base_acc
                fair_change = (abs(base_fair) - abs(top_model[base_fairness_metric])) / abs(base_fair)
                
                # Add to comparison data
                comparison_data.append({
                    'Model': top_model['Model'],
                    'Model_Type': f"{method_name} ({fairness_name})",
                    'Base_Model': 'TSCFM Base',
                    'Acc': top_model['Acc'],
                    'Fairness': abs(top_model[base_fairness_metric]),
                    'Acc_Change': acc_change,
                    'Fair_Change': fair_change,
                    'Method': cf_method,
                    'Fairness_Type': fairness_code,
                    'Is_TSCFM': True
                })
    
        # Process ALL baseline fair models (not just those that improve fairness)
        for fair_model in baseline_fair_models:
            if fair_model not in fair_model_metrics:
                continue
                
            fair_info = fair_model_metrics[fair_model]
            base_name = fair_info['base_model']
            
            # Only process if we have the base model metrics
            if base_name not in base_model_metrics:
                continue
                
            base_info = base_model_metrics[base_name]
            
            # Calculate changes with correct percentage formula
            acc_change = (fair_info['Acc'] - base_info['Acc']) / base_info['Acc']
            
            # Check if we have the fairness metric
            if base_fairness_metric in fair_info and base_fairness_metric in base_info:
                fair_change = (abs(base_info[base_fairness_metric]) - abs(fair_info[base_fairness_metric])) / abs(base_info[base_fairness_metric])
                
                # Add to comparison data
                comparison_data.append({
                    'Model': fair_model,
                    'Model_Type': f"Baseline Fair ({base_name})",
                    'Base_Model': base_name,
                    'Acc': fair_info['Acc'],
                    'Fairness': abs(fair_info[base_fairness_metric]),
                    'Acc_Change': acc_change,
                    'Fair_Change': fair_change,
                    'Method': 'Baseline',
                    'Fairness_Type': fairness_code,
                    'Is_TSCFM': False
                })
    
    # Convert comparison data to DataFrame
    if not comparison_data:
        print("Warning: No comparison data generated")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save the comparison data for reference
    output_csv_path = os.path.join(output_dir, f"{file_prefix}_top_models_comparison.csv")
    comparison_df.to_csv(output_csv_path, index=False)
    print(f"Saved comparison data to {output_csv_path}")
    
    # Create visualizations
    # 1. Scatter plot of accuracy change vs fairness change
    plt.figure(figsize=(12, 10))
    
    # Define a color mapping for model types
    model_type_colors = {
        'G': method_colors['G'],
        'S': method_colors['S'],
        'M': method_colors['M'],
        'Baseline': 'gray'
    }
    
    # Plot each point with appropriate color and marker
    for _, row in comparison_df.iterrows():
        color = model_type_colors.get(row['Method'], 'black')
        marker = 'o' if row['Is_TSCFM'] else 'x'
        label = row['Model_Type'] if row['Is_TSCFM'] or 'Baseline' in row['Model_Type'] else None
        
        plt.scatter(
            row['Acc_Change'] * 100, # Convert to percentage
            row['Fair_Change'] * 100, # Convert to percentage  
            color=color,
            marker=marker,
            s=150,
            alpha=0.7,
            label=label
        )
        
        # Add model name text
        plt.annotate(
            row['Model'],
            (row['Acc_Change'] * 100, row['Fair_Change'] * 100),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            color=color
        )
    
    # Add quadrant lines
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Get the bounds for quadrant labels
    x_range = max(comparison_df['Acc_Change'] * 100) - min(comparison_df['Acc_Change'] * 100)
    y_range = max(comparison_df['Fair_Change'] * 100) - min(comparison_df['Fair_Change'] * 100)
    
    # Add quadrant labels
    plt.text(
        max(comparison_df['Acc_Change'] * 100) - x_range*0.25,
        max(comparison_df['Fair_Change'] * 100) - y_range*0.25,
        "Better Accuracy\nBetter Fairness",
        ha='center',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'),
        color='green'
    )
    
    plt.text(
        min(comparison_df['Acc_Change'] * 100) + x_range*0.25,
        max(comparison_df['Fair_Change'] * 100) - y_range*0.25,
        "Worse Accuracy\nBetter Fairness",
        ha='center',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'),
        color='orange'
    )
    
    plt.text(
        max(comparison_df['Acc_Change'] * 100) - x_range*0.25,
        min(comparison_df['Fair_Change'] * 100) + y_range*0.25,
        "Better Accuracy\nWorse Fairness",
        ha='center',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'),
        color='orange'
    )
    
    plt.text(
        min(comparison_df['Acc_Change'] * 100) + x_range*0.25,
        min(comparison_df['Fair_Change'] * 100) + y_range*0.25,
        "Worse Accuracy\nWorse Fairness",
        ha='center',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'),
        color='red'
    )
    
    # Set labels and title
    plt.xlabel('Change in Accuracy (%)', fontsize=14)
    plt.ylabel('Change in Fairness (%)', fontsize=14)
    plt.title('Comparison of Top Models: Percentage Changes from Base Models', fontsize=16)
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Handle legend with duplicates removed
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{file_prefix}_fairness_accuracy_changes.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fairness-accuracy changes plot to {output_path}")
    
    # 2. Create a grouped bar chart for easier comparison
    plt.figure(figsize=(15, 10))
    
    # Prepare data for grouped bar chart
    # Focus on one fairness metric at a time
    for fairness_code in fairness_info.keys():
        fairness_subset = comparison_df[comparison_df['Fairness_Type'] == fairness_code].copy()
        
        if len(fairness_subset) == 0:
            continue
        
        # Create the bar chart
        plt.figure(figsize=(14, 8))
        
        # Sort models by fairness (lowest fairness first)
        fairness_subset = fairness_subset.sort_values('Fairness')
        
        # Set positions for bars
        bar_width = 0.35
        index = np.arange(len(fairness_subset))
        
        # Plot accuracy changes
        plt.bar(
            index - bar_width/2,
            fairness_subset['Acc_Change'] * 100,  # Convert to percentage
            bar_width,
            alpha=0.7,
            color=[model_type_colors.get(row['Method'], 'black') for _, row in fairness_subset.iterrows()],
            label='Accuracy Change (%)'
        )
        
        # Plot fairness changes
        plt.bar(
            index + bar_width/2,
            fairness_subset['Fair_Change'] * 100,  # Convert to percentage
            bar_width,
            alpha=0.7,
            color=[mcolors.to_rgba(model_type_colors.get(row['Method'], 'black'), 0.5) for _, row in fairness_subset.iterrows()],
            label='Fairness Change (%)',
            hatch='//'
        )
        
        # Set labels and title
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Percentage Change from Base Model', fontsize=14)
        plt.title(f'Changes in Accuracy and Fairness ({fairness_info[fairness_code]["name"]})', fontsize=16)
        
        # Set x-ticks
        plt.xticks(
            index,
            [f"{row['Model']}\n({row['Model_Type']})" for _, row in fairness_subset.iterrows()],
            rotation=45,
            ha='right'
        )
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        # Add a horizontal line at 0
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add legend
        plt.legend()
        
        # Add value labels on bars
        for i, (_, row) in enumerate(fairness_subset.iterrows()):
            plt.text(
                i - bar_width/2,
                row['Acc_Change'] * 100 + (5 if row['Acc_Change'] > 0 else -10),
                f"{row['Acc_Change']*100:.1f}%",
                ha='center',
                va='bottom' if row['Acc_Change'] > 0 else 'top',
                fontsize=9
            )
            
            plt.text(
                i + bar_width/2,
                row['Fair_Change'] * 100 + (5 if row['Fair_Change'] > 0 else -10),
                f"{row['Fair_Change']*100:.1f}%",
                ha='center',
                va='bottom' if row['Fair_Change'] > 0 else 'top',
                fontsize=9
            )
        
        # Save the plot
        output_path = os.path.join(output_dir, f"{file_prefix}_{fairness_code}_changes_barchart.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {fairness_code} changes bar chart to {output_path}")
    
    # 3. Create a direct fairness vs accuracy plot
    for fairness_code in fairness_info.keys():
        fairness_subset = comparison_df[comparison_df['Fairness_Type'] == fairness_code].copy()
        
        if len(fairness_subset) == 0:
            continue
        
        plt.figure(figsize=(12, 10))
        
        # Identify TSCFM and baseline models
        tscfm_models = fairness_subset[fairness_subset['Is_TSCFM']]
        baseline_models = fairness_subset[~fairness_subset['Is_TSCFM']]
        
        # Plot baseline models
        if len(baseline_models) > 0:
            plt.scatter(
                baseline_models['Acc'], 
                baseline_models['Fairness'],
                color='gray',
                marker='x',
                s=100,
                alpha=0.7,
                label='Baseline Fair Models'
            )
            
            # Add model name annotations
            for _, row in baseline_models.iterrows():
                plt.annotate(
                    row['Model'],
                    (row['Acc'], row['Fairness']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color='gray'
                )
        
        # Plot TSCFM models by method
        for method, color in method_colors.items():
            method_models = tscfm_models[tscfm_models['Method'] == method]
            
            if len(method_models) > 0:
                method_name = method_names.get(method, method)
                
                plt.scatter(
                    method_models['Acc'], 
                    method_models['Fairness'],
                    color=color,
                    marker='o',
                    s=120,
                    alpha=0.8,
                    label=f"{method_name} Models"
                )
                
                # Add model name annotations
                for _, row in method_models.iterrows():
                    plt.annotate(
                        row['Model'],
                        (row['Acc'], row['Fairness']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        color=color
                    )
        
        # Set labels and title
        plt.xlabel('Accuracy', fontsize=14)
        plt.ylabel(f"Fairness ({fairness_info[fairness_code]['name']})", fontsize=14)
        plt.title(f'Fairness-Accuracy Trade-off ({fairness_info[fairness_code]["name"]})', fontsize=16)
        
        # Add grid
        plt.grid(alpha=0.3)
        
        # Add the base model point
        plt.scatter(
            tscfm_base_acc, 
            abs(tscfm_base_fairness[fairness_info[fairness_code]['base_metric']]),
            color='black',
            marker='*',
            s=200,
            label='Base Model'
        )
        
        # Add a text label for the base model
        plt.annotate(
            'TSCFM Base',
            (tscfm_base_acc, abs(tscfm_base_fairness[fairness_info[fairness_code]['base_metric']])),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )
        
        # Handle legend with duplicates removed
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        # Save the plot
        output_path = os.path.join(output_dir, f"{file_prefix}_{fairness_code}_tradeoff.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {fairness_code} trade-off plot to {output_path}")


# Example usage
if __name__ == "__main__":
    # Load results
    # results_df = pd.read_csv(r"path_to_file")


    # Define base metrics for TSCFM
    #german
    # tscfm_base_acc = 0.7533  # Example value - replace with your actual base accuracy
    # tscfm_base_fairness = {  # Example values - replace with your actual base fairness metrics
    #     'SP': 0.032475, 
    #     'EO': 0.040816, 
    #     'EOd': 0.104612
    # }
    #pakdd
    # tscfm_base_acc = 0.73973  # Base accuracy value
    # tscfm_base_fairness = {  # Base fairness values
    #     'SP': 0.001065, 
    #     'EO': 0.00086, 
    #     'EOd': 0.003263
    # }
    #card credit
    tscfm_base_acc = 0.8172  # Base accuracy value
    tscfm_base_fairness = {  # Base fairness values
        'SP': 0.006786, 
        'EO': 0.028098, 
        'EOd': 0.033806
    }
    
    # Create top model comparison visualizations
    visualize_top_models_comparison(
        results_df, 
        "output_directory", 
        "card",
        tscfm_base_acc,
        tscfm_base_fairness
    )
