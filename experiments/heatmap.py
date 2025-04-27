import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from typing import List, Dict, Tuple, Optional

def extract_parameters_from_model_name(model_name: str) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[int]]:
    """
    Extract parameters from model name with format like "G-DP-13.0-4"
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (cf_method, fairness_type, adjustment_strength, amplification_factor)
    """
    if not isinstance(model_name, str):
        return None, None, None, None
    
    # First, extract the method (G, S, M)
    cf_method = None
    if model_name.startswith('G-'):
        cf_method = 'G'
    elif model_name.startswith('S-'):
        cf_method = 'S'
    elif model_name.startswith('M-'):
        cf_method = 'M'
    else:
        return None, None, None, None
    
    # Match the pattern more flexibly: [G/S/M]-[fairness_type]-[adjustment_strength]-[amplification_factor]
    pattern = r'^[GSM]-([^-]+)-([0-9.]+)-([0-9]+)'
    match = re.match(pattern, model_name)
    
    if match:
        fairness_type, adj_str, amp_factor = match.groups()
        try:
            adjustment_strength = float(adj_str)
            amplification_factor = int(amp_factor)
            return cf_method, fairness_type, adjustment_strength, amplification_factor
        except (ValueError, TypeError):
            pass
    
    return cf_method, None, None, None

def print_model_names_sample(df, column='Model', n=10):
    """
    Print a sample of model names to help debug pattern matching issues.
    
    Args:
        df: DataFrame containing the models
        column: Column name containing model names
        n: Number of samples to print
    """
    print(f"\nSample of {min(n, len(df))} model names:")
    for model in df[column].sample(min(n, len(df))).values:
        print(f"  {model}")

def create_parameter_heatmaps(results_df: pd.DataFrame, 
                            output_dir: str,
                            file_prefix: str = 'german',
                            debug: bool = True) -> None:
    """
    Create heat maps showing how adjustment strength and amplification factor 
    affect fairness metrics.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save output files
        file_prefix: Prefix for output files
        debug: Whether to print debug information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify TSCFM models by prefix rather than regex
    results_df['is_tscfm'] = results_df['Model'].apply(
        lambda x: isinstance(x, str) and x.startswith(('G-', 'S-', 'M-'))
    )
    
    # Get only TSCFM models
    tscfm_df = results_df[results_df['is_tscfm']].copy()
    
    if len(tscfm_df) == 0:
        print("Warning: No TSCFM models found")
        if debug:
            print_model_names_sample(results_df)
        return
    
    if debug:
        print(f"Found {len(tscfm_df)} TSCFM models")
        print_model_names_sample(tscfm_df)
    
    # Add columns for absolute values of fairness metrics if not already present
    for metric in ['SP', 'EO', 'EOd']:
        if f"{metric}_abs" not in tscfm_df.columns:
            tscfm_df[f"{metric}_abs"] = tscfm_df[metric].abs()
    
    # Extract parameters from model names
    params = tscfm_df['Model'].apply(extract_parameters_from_model_name)
    tscfm_df['cf_method'] = params.apply(lambda x: x[0])
    tscfm_df['fairness_type'] = params.apply(lambda x: x[1])
    tscfm_df['adjustment_strength'] = params.apply(lambda x: x[2])
    tscfm_df['amplification_factor'] = params.apply(lambda x: x[3])
    
    # Check the extracted values
    if debug:
        print("\nExtracted parameters summary:")
        print(f"Counterfactual methods: {tscfm_df['cf_method'].unique().tolist()}")
        print(f"Fairness types: {tscfm_df['fairness_type'].unique().tolist()}")
        print(f"Adjustment strengths: {tscfm_df['adjustment_strength'].unique().tolist()}")
        print(f"Amplification factors: {tscfm_df['amplification_factor'].unique().tolist()}")
        
        # Count the null values
        print("\nNull values count in extracted parameters:")
        for col in ['cf_method', 'fairness_type', 'adjustment_strength', 'amplification_factor']:
            print(f"{col}: {tscfm_df[col].isna().sum()} null values")
    
    # Define fairness metrics to use with their corresponding model pattern
    fairness_metrics = [
        ('SP_abs', 'Demographic Parity Difference', 'DP'),
        ('EO_abs', 'Equal Opportunity Difference', 'EO'),
        ('EOd_abs', 'Equalized Odds Difference', 'EOd')
    ]
    
    # Try additional patterns that might be in the data
    all_fairness_patterns = ['DP', 'EO', 'EOd', 'SP', 'Fairness', 'Fair']
    
    # Add any unique fairness types found in the data
    found_fairness_types = [ft for ft in tscfm_df['fairness_type'].unique() if ft is not None]
    all_fairness_patterns.extend([ft for ft in found_fairness_types if ft not in all_fairness_patterns])
    
    # Define counterfactual methods
    cf_methods = {
        'G': 'Generative',
        'S': 'Structural Equation',
        'M': 'Matching'
    }
    
    # Process each fairness metric
    for fair_metric, fair_name, model_pattern in fairness_metrics:
        # Try to find models optimized for this fairness metric, checking multiple possible patterns
        for pattern in all_fairness_patterns:
            # Filter to models that might be optimizing for this metric
            if pattern == model_pattern:
                specific_df = tscfm_df[tscfm_df['fairness_type'] == pattern].copy()
                if len(specific_df) > 0:
                    print(f"Found {len(specific_df)} models optimizing for {fair_name} with pattern '{pattern}'")
                    break
        else:
            # If no models found with exact pattern, try a more relaxed approach - just use all models
            print(f"Warning: No specific models found optimizing for {fair_name}")
            print(f"Using all TSCFM models for {fair_name} analysis")
            specific_df = tscfm_df.copy()
        
        for cf_method, cf_name in cf_methods.items():
            # Filter to models with this counterfactual method
            method_df = specific_df[specific_df['cf_method'] == cf_method].copy()
            
            if len(method_df) == 0:
                print(f"Warning: No models found for {cf_name} method")
                continue
            
            # Filter out rows with missing parameter values
            method_df = method_df.dropna(subset=['adjustment_strength', 'amplification_factor'])
            
            if len(method_df) == 0:
                print(f"Warning: No models found for {cf_name} method with valid parameter values")
                continue
            
            # Get unique values of adjustment strength and amplification factor
            adj_strengths = sorted(method_df['adjustment_strength'].unique())
            amp_factors = sorted(method_df['amplification_factor'].unique())
            
            if len(adj_strengths) <= 1 or len(amp_factors) <= 1:
                print(f"Warning: Not enough parameter variation for {cf_name} method")
                print(f"Adjustment strengths: {adj_strengths}")
                print(f"Amplification factors: {amp_factors}")
                continue
            
            print(f"Creating heatmaps for {cf_name} method with {fair_name}")
            print(f"Number of models: {len(method_df)}")
            print(f"Adjustment strengths: {adj_strengths}")
            print(f"Amplification factors: {amp_factors}")
            
            # Create pivot table for heatmap - fairness metric
            pivot_fair = method_df.pivot_table(
                index='amplification_factor', 
                columns='adjustment_strength',
                values=fair_metric,
                aggfunc='mean'
            )
            
            # Create pivot table for heatmap - accuracy
            pivot_acc = method_df.pivot_table(
                index='amplification_factor', 
                columns='adjustment_strength',
                values='Acc',
                aggfunc='mean'
            )
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Fairness heatmap
            sns.heatmap(
                pivot_fair, 
                annot=True, 
                fmt=".4f", 
                cmap="viridis_r",  # Reversed colormap so darker is better (lower fairness metric)
                ax=ax1
            )
            ax1.set_title(f"{fair_name} by Parameters\n({cf_name} Method)", fontsize=14)
            ax1.set_xlabel("Adjustment Strength", fontsize=12)
            ax1.set_ylabel("Amplification Factor", fontsize=12)
            
            # Accuracy heatmap
            sns.heatmap(
                pivot_acc, 
                annot=True, 
                fmt=".4f", 
                cmap="viridis",  # Regular colormap so darker is worse (lower accuracy)
                ax=ax2
            )
            ax2.set_title(f"Accuracy by Parameters\n({cf_name} Method)", fontsize=14)
            ax2.set_xlabel("Adjustment Strength", fontsize=12)
            ax2.set_ylabel("Amplification Factor", fontsize=12)
            
            # Add a suptitle
            plt.suptitle(f"Effect of Parameters on {fair_name} and Accuracy for {cf_name} Method", fontsize=16)
            
            # Save the figure
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            
            output_path = os.path.join(
                output_dir, 
                f"{file_prefix}_{fair_metric.replace('_abs', '')}_{cf_method}_parameter_heatmap.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved parameter heatmap to {output_path}")
            
            # Create a third plot - combined metric (balanced score)
            # Normalize fairness and accuracy to 0-1 range
            norm_fair = (pivot_fair - pivot_fair.min().min()) / (pivot_fair.max().max() - pivot_fair.min().min())
            norm_acc = (pivot_acc - pivot_acc.min().min()) / (pivot_acc.max().max() - pivot_acc.min().min())
            
            # Invert fairness normalization since lower is better
            norm_fair = 1 - norm_fair
            
            # Calculate balanced score (harmonic mean)
            balanced_score = 2 * norm_fair * norm_acc / (norm_fair + norm_acc)
            
            # Create figure for balanced score
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                balanced_score, 
                annot=True, 
                fmt=".2f", 
                cmap="viridis",
                vmin=0, vmax=1
            )
            plt.title(f"Balanced Score by Parameters\n({cf_name} Method, {fair_name})", fontsize=14)
            plt.xlabel("Adjustment Strength", fontsize=12)
            plt.ylabel("Amplification Factor", fontsize=12)
            
            # Save the figure
            plt.tight_layout()
            
            output_path = os.path.join(
                output_dir, 
                f"{file_prefix}_{fair_metric.replace('_abs', '')}_{cf_method}_balanced_heatmap.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved balanced score heatmap to {output_path}")
            
            # Create contour plot with fairness and accuracy
            plt.figure(figsize=(10, 8))
            
            # Create a meshgrid for contour plotting
            X, Y = np.meshgrid(adj_strengths, amp_factors)
            
            
            # 3D surface plot
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create a surface plot
                surf = ax.plot_surface(
                    X, Y, 
                    pivot_fair.values,
                    cmap='viridis_r',
                    edgecolor='none',
                    alpha=0.7
                )
                
                # Add scatter points for actual models
                ax.scatter(
                    method_df['adjustment_strength'],
                    method_df['amplification_factor'],
                    method_df[fair_metric],
                    c=method_df[fair_metric],
                    cmap='viridis_r',
                    s=50,
                    edgecolor='black'
                )
                
                # Add labels
                ax.set_xlabel("Adjustment Strength", fontsize=12)
                ax.set_ylabel("Amplification Factor", fontsize=12)
                ax.set_zlabel(fair_name, fontsize=12)
                ax.set_title(f"3D Surface Plot of {fair_name}\n({cf_name} Method)", fontsize=14)
                
                # Add a color bar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=fair_name)
                
                # Save the figure
                plt.tight_layout()
                
                output_path = os.path.join(
                    output_dir, 
                    f"{file_prefix}_{fair_metric.replace('_abs', '')}_{cf_method}_surface3d.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved 3D surface plot to {output_path}")
            except Exception as e:
                print(f"Could not create 3D surface plot: {e}")

# Example usage
if __name__ == "__main__":
    # Load results
    results_df = pd.read_csv(r"path_to_file")

    create_parameter_heatmaps(results_df, "output_directory", "card")
