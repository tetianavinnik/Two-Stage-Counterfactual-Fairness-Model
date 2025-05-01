import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.colors as mcolors
from scipy import stats

def create_fairness_cross_impact_heatmap(results_df: pd.DataFrame,
                                        output_dir: str,
                                        file_prefix: str = 'german',
                                        tscfm_base_acc: float = 0.7533,
                                        tscfm_base_fairness: Dict[str, float] = None) -> None:
    """
    Create a correlation heatmap showing how improvements in the target fairness metric 
    correlate with changes in other fairness metrics.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save output files
        file_prefix: Prefix for output files
        tscfm_base_acc: Base accuracy for calculating improvements
        tscfm_base_fairness: Dictionary with base fairness values for calculating improvements
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default fairness values if not provided
    if tscfm_base_fairness is None:
        tscfm_base_fairness = {'SP': 0.032475, 'EO': 0.040816, 'EOd': 0.104612}
    
    # Identify TSCFM models by prefix
    results_df['is_tscfm'] = results_df['Model'].apply(
        lambda x: isinstance(x, str) and x.startswith(('G-', 'S-', 'M-'))
    )
    
    # Get only TSCFM models
    tscfm_df = results_df[results_df['is_tscfm']].copy()
    
    if len(tscfm_df) == 0:
        print("Warning: No TSCFM models found")
        return
    
    # Extract fairness constraint type from model name
    tscfm_df['fairness_type'] = tscfm_df['Model'].apply(
        lambda x: re.search(r'-([^-]+)-', x).group(1) if isinstance(x, str) and re.search(r'-([^-]+)-', x) else None
    )
    
    # Map fairness type codes to full names
    fairness_names = {
        'DP': 'Demographic Parity',
        'EO': 'Equal Opportunity',
        'EOd': 'Equalized Odds'
    }
    
    # Replace fairness type codes with full names
    tscfm_df['fairness_name'] = tscfm_df['fairness_type'].map(fairness_names)
    
    # Add columns for absolute values of fairness metrics if not already present
    for metric in ['SP', 'EO', 'EOd']:
        if f"{metric}_abs" not in tscfm_df.columns:
            tscfm_df[f"{metric}_abs"] = tscfm_df[metric].abs()
    
    # Define fairness metrics mapping between name, absolute metric, and base metric
    fairness_metrics = [
        ('DP', 'Demographic Parity', 'SP_abs', 'SP'),
        ('EO', 'Equal Opportunity', 'EO_abs', 'EO'),
        ('EOd', 'Equalized Odds', 'EOd_abs', 'EOd')
    ]
    
    # Calculate improvement percentages for each fairness metric
    for _, _, abs_metric, base_metric in fairness_metrics:
        # Calculate percentage improvement: negative values mean improvement (less bias)
        base_value = abs(tscfm_base_fairness[base_metric])
        tscfm_df[f'{abs_metric}_improvement'] = (tscfm_df[abs_metric] - base_value) / base_value * 100
    
    # Calculate accuracy improvement percentage
    tscfm_df['Acc_improvement'] = (tscfm_df['Acc'] - tscfm_base_acc) / tscfm_base_acc * 100
    
    # Create data for cross-impact correlation heatmap
    correlation_data = []
    
    # For each fairness type, collect correlations between its improvements and other fairness metrics
    for fairness_code, fairness_name, abs_metric, base_metric in fairness_metrics:
        # Get models optimizing for this fairness type
        type_models = tscfm_df[tscfm_df['fairness_type'] == fairness_code].copy()
        
        if len(type_models) == 0:
            print(f"Warning: No models found optimizing for {fairness_name}")
            continue
        
        # Filter to models that improved on the targeted fairness metric
        improved_models = type_models[type_models[f'{abs_metric}_improvement'] < 0]
        
        if len(improved_models) == 0:
            print(f"Warning: No models found that improved {fairness_name}")
            continue
        
        print(f"Found {len(improved_models)} models optimizing for and improving {fairness_name}")
        
        # Calculate correlations between improvements in all fairness metrics
        improvements = [f'{m}_abs_improvement' for _, _, m, _ in fairness_metrics]
        
        correlation_row = {'Optimization Target': fairness_name}
        
        # For each other fairness metric, calculate correlation with the targeted one
        for other_code, other_name, other_abs, _ in fairness_metrics:
            if fairness_code == other_code:
                # Skip self-correlation (always 1.0)
                continue
                
            # Calculate correlation between improvement in targeted metric and this metric
            correlation, p_value = stats.pearsonr(
                improved_models[f'{abs_metric}_improvement'],
                improved_models[f'{other_abs}_improvement']
            )
            
            # Add correlation to the data
            correlation_row[other_name] = correlation
            
            # Add significance marker
            sig = ""
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
                
            correlation_row[f'{other_name}_sig'] = sig
            correlation_row[f'{other_name}_p'] = p_value
            
            print(f"  Correlation with {other_name}: {correlation:.4f} (p={p_value:.4f})")
        
        # Add the row to our data
        correlation_data.append(correlation_row)
    
    # Create the correlation dataframe
    if not correlation_data:
        print("Warning: No correlation data available")
        return
        
    corr_df = pd.DataFrame(correlation_data)
    
    # Prepare data for the heatmap
    fairness_types = [name for _, name, _, _ in fairness_metrics]
    matrix_data = np.zeros((len(fairness_types), len(fairness_types)))
    p_values = np.ones((len(fairness_types), len(fairness_types)))
    
    # Fill the matrix with correlation values
    for i, row_name in enumerate(fairness_types):
        # Diagonal is 1.0 (self-correlation)
        matrix_data[i, i] = 1.0
        
        # Find the row for this optimization target
        row_data = corr_df[corr_df['Optimization Target'] == row_name]
        
        if len(row_data) == 0:
            continue
            
        # Fill in correlations with other metrics
        for j, col_name in enumerate(fairness_types):
            if i != j:  # Skip diagonal
                if col_name in row_data.columns:
                    matrix_data[i, j] = row_data[col_name].values[0]
                    
                    # Also get p-value if available
                    if f'{col_name}_p' in row_data.columns:
                        p_values[i, j] = row_data[f'{col_name}_p'].values[0]
    
    plt.figure(figsize=(12, 10))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Create the heatmap
    sns.heatmap(
        matrix_data,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1, vmax=1,
        square=True,
        xticklabels=fairness_types,
        yticklabels=fairness_types,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    # Add significance markers
    for i in range(len(fairness_types)):
        for j in range(len(fairness_types)):
            if i != j:  # Skip diagonal
                p = p_values[i, j]
                sig = ""
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                    
                if sig:
                    # Move stars up significantly (change i + 0.5 to i + 0.2)
                    # This places them near the top of the cell
                    plt.text(j + 0.5, i + 0.2, sig,
                            ha='center', va='center',
                            color='black', fontsize=10, fontweight='bold')
    
    # Add labels and title
    plt.title('Cross-Impact: How Optimizing for One Fairness Metric Affects Others', fontsize=16)
    plt.xlabel('Fairness Metric', fontsize=14)
    plt.ylabel('Optimization Target', fontsize=14)
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
                "Rows: Models optimizing for and improving specific fairness metric\n" +
                "Columns: Correlation with improvement in other fairness metrics\n" +
                "Significance: * p<0.05, ** p<0.01, *** p<0.001",
                ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Save the heatmap
    output_path = os.path.join(output_dir, f"{file_prefix}_fairness_cross_impact_correlation.png")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fairness cross-impact correlation heatmap to {output_path}")

# Example usage
if __name__ == "__main__":
    # Load results
    results_df = pd.read_csv(r"path_to_results_file")

    #german
    # Specify base model metrics
    # tscfm_base_acc = 0.7533  # Base accuracy value
    # tscfm_base_fairness = {  # Base fairness values
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

    #credit card
    tscfm_base_acc = 0.8172  # Base accuracy value
    tscfm_base_fairness = {  # Base fairness values
        'SP': 0.006786, 
        'EO': 0.028098, 
        'EOd': 0.033806
    }
    
    # Create cross-impact correlation heatmap
    create_fairness_cross_impact_heatmap(
        results_df, 
        "output_directory", 
        "card",
        tscfm_base_acc,
        tscfm_base_fairness
    )
