# visualizations/fairness.py - Fairness visualization functions

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from config import METRIC_DISPLAY_NAMES, PRIMARY_FAIRNESS_METRICS

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_probability_distributions(original_probs, fair_probs, protected_attr, 
                                   output_path=None, bins=20):
    """
    Plot histograms of prediction probabilities before and after fairness adjustment
    
    Args:
        original_probs: Original prediction probabilities
        fair_probs: Fair prediction probabilities
        protected_attr: Protected attribute values (0/1)
        output_path: Path to save the plot (if None, just display)
        bins: Number of histogram bins
    
    Returns:
        Matplotlib figure
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data by group
    group0_orig = original_probs[protected_attr == 0]
    group1_orig = original_probs[protected_attr == 1]
    
    group0_fair = fair_probs[protected_attr == 0]
    group1_fair = fair_probs[protected_attr == 1]
    
    # Calculate group means
    group0_orig_mean = np.mean(group0_orig)
    group1_orig_mean = np.mean(group1_orig)
    group0_fair_mean = np.mean(group0_fair)
    group1_fair_mean = np.mean(group1_fair)
    
    # Calculate demographic parity difference
    dpd_orig = np.abs(group0_orig_mean - group1_orig_mean)
    dpd_fair = np.abs(group0_fair_mean - group1_fair_mean)
    
    # Plot original probabilities
    ax1.hist(group0_orig, bins=bins, alpha=0.7, label='Group 0', color='blue')
    ax1.hist(group1_orig, bins=bins, alpha=0.7, label='Group 1', color='red')
    
    # Add vertical lines for means
    ax1.axvline(group0_orig_mean, color='blue', linestyle='--', linewidth=2)
    ax1.axvline(group1_orig_mean, color='red', linestyle='--', linewidth=2)
    
    # Plot fair probabilities
    ax2.hist(group0_fair, bins=bins, alpha=0.7, label='Group 0', color='blue')
    ax2.hist(group1_fair, bins=bins, alpha=0.7, label='Group 1', color='red')
    
    # Add vertical lines for means
    ax2.axvline(group0_fair_mean, color='blue', linestyle='--', linewidth=2)
    ax2.axvline(group1_fair_mean, color='red', linestyle='--', linewidth=2)
    
    # Add labels and formatting
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Original Predictions')
    ax1.legend()
    ax1.text(0.05, 0.95, f'Mean G0: {group0_orig_mean:.3f}\nMean G1: {group1_orig_mean:.3f}\nDPD: {dpd_orig:.3f}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Fair Predictions')
    ax2.legend()
    ax2.text(0.05, 0.95, f'Mean G0: {group0_fair_mean:.3f}\nMean G1: {group1_fair_mean:.3f}\nDPD: {dpd_fair:.3f}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Calculate improvement percentage
    if dpd_orig > 0:
        improvement = (dpd_orig - dpd_fair) / dpd_orig * 100
        plt.suptitle(f'Prediction Probability Distributions (DPD Improvement: {improvement:.1f}%)', fontsize=16)
    else:
        plt.suptitle('Prediction Probability Distributions', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_roc_curves_by_group(y_true, y_score, protected_attr, 
                            output_path=None, title='ROC Curves by Group'):
    """
    Plot ROC curves for each protected attribute group
    
    Args:
        y_true: True class labels
        y_score: Prediction probabilities
        protected_attr: Protected attribute values (0/1)
        output_path: Path to save the plot (if None, just display)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get indices for each group
    group0_idx = (protected_attr == 0)
    group1_idx = (protected_attr == 1)
    
    # Compute ROC curve for the entire dataset
    fpr_all, tpr_all, _ = roc_curve(y_true, y_score)
    roc_auc_all = auc(fpr_all, tpr_all)
    
    # Compute ROC curve for group 0
    fpr_0, tpr_0, _ = roc_curve(y_true[group0_idx], y_score[group0_idx])
    roc_auc_0 = auc(fpr_0, tpr_0)
    
    # Compute ROC curve for group 1
    fpr_1, tpr_1, _ = roc_curve(y_true[group1_idx], y_score[group1_idx])
    roc_auc_1 = auc(fpr_1, tpr_1)
    
    # Calculate ABROCA (area between ROC curves)
    # Interpolate ROC curves to the same set of points
    combined_fpr = np.sort(np.unique(np.concatenate([fpr_0, fpr_1])))
    tpr_0_interp = np.interp(combined_fpr, fpr_0, tpr_0)
    tpr_1_interp = np.interp(combined_fpr, fpr_1, tpr_1)
    
    # Calculate area between curves
    abroca = np.abs(np.trapz(tpr_0_interp, combined_fpr) - np.trapz(tpr_1_interp, combined_fpr))
    
    # Plot ROC curves
    ax.plot(fpr_all, tpr_all, 'k-', lw=2, label=f'All (AUC = {roc_auc_all:.3f})')
    ax.plot(fpr_0, tpr_0, 'b-', lw=2, label=f'Group 0 (AUC = {roc_auc_0:.3f})')
    ax.plot(fpr_1, tpr_1, 'r-', lw=2, label=f'Group 1 (AUC = {roc_auc_1:.3f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Add labels and formatting
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title} (ABROCA = {abroca:.4f})')
    ax.legend(loc='lower right')
    
    # Add the exact ABROCA value
    ax.text(0.5, 0.1, f'ABROCA = {abroca:.4f}', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_roc_curves_comparison(y_true, original_scores, fair_scores, protected_attr, 
                              output_path=None):
    """
    Plot ROC curves for original and fair models
    
    Args:
        y_true: True class labels
        original_scores: Original prediction probabilities
        fair_scores: Fair prediction probabilities
        protected_attr: Protected attribute values (0/1)
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get indices for each group
    group0_idx = (protected_attr == 0)
    group1_idx = (protected_attr == 1)
    
    # Compute ROC curves for original scores
    fpr_0_orig, tpr_0_orig, _ = roc_curve(y_true[group0_idx], original_scores[group0_idx])
    roc_auc_0_orig = auc(fpr_0_orig, tpr_0_orig)
    
    fpr_1_orig, tpr_1_orig, _ = roc_curve(y_true[group1_idx], original_scores[group1_idx])
    roc_auc_1_orig = auc(fpr_1_orig, tpr_1_orig)
    
    # Compute ROC curves for fair scores
    fpr_0_fair, tpr_0_fair, _ = roc_curve(y_true[group0_idx], fair_scores[group0_idx])
    roc_auc_0_fair = auc(fpr_0_fair, tpr_0_fair)
    
    fpr_1_fair, tpr_1_fair, _ = roc_curve(y_true[group1_idx], fair_scores[group1_idx])
    roc_auc_1_fair = auc(fpr_1_fair, tpr_1_fair)
    
    # Calculate ABROCA for original and fair scores
    # Interpolate original ROC curves
    orig_combined_fpr = np.sort(np.unique(np.concatenate([fpr_0_orig, fpr_1_orig])))
    tpr_0_orig_interp = np.interp(orig_combined_fpr, fpr_0_orig, tpr_0_orig)
    tpr_1_orig_interp = np.interp(orig_combined_fpr, fpr_1_orig, tpr_1_orig)
    abroca_orig = np.abs(np.trapz(tpr_0_orig_interp, orig_combined_fpr) - 
                          np.trapz(tpr_1_orig_interp, orig_combined_fpr))
    
    # Interpolate fair ROC curves
    fair_combined_fpr = np.sort(np.unique(np.concatenate([fpr_0_fair, fpr_1_fair])))
    tpr_0_fair_interp = np.interp(fair_combined_fpr, fpr_0_fair, tpr_0_fair)
    tpr_1_fair_interp = np.interp(fair_combined_fpr, fpr_1_fair, tpr_1_fair)
    abroca_fair = np.abs(np.trapz(tpr_0_fair_interp, fair_combined_fpr) - 
                         np.trapz(tpr_1_fair_interp, fair_combined_fpr))
    
    # Plot original ROC curves
    ax1.plot(fpr_0_orig, tpr_0_orig, 'b-', lw=2, label=f'Group 0 (AUC = {roc_auc_0_orig:.3f})')
    ax1.plot(fpr_1_orig, tpr_1_orig, 'r-', lw=2, label=f'Group 1 (AUC = {roc_auc_1_orig:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Plot fair ROC curves
    ax2.plot(fpr_0_fair, tpr_0_fair, 'b-', lw=2, label=f'Group 0 (AUC = {roc_auc_0_fair:.3f})')
    ax2.plot(fpr_1_fair, tpr_1_fair, 'r-', lw=2, label=f'Group 1 (AUC = {roc_auc_1_fair:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Add labels and formatting
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Original Model (ABROCA = {abroca_orig:.4f})')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'Fair Model (ABROCA = {abroca_fair:.4f})')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Calculate improvement percentage
    if abroca_orig > 0:
        improvement = (abroca_orig - abroca_fair) / abroca_orig * 100
        plt.suptitle(f'ROC Curves Comparison (ABROCA Improvement: {improvement:.1f}%)', fontsize=16)
    else:
        plt.suptitle('ROC Curves Comparison', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_fairness_metrics_radar(baseline_metrics, fair_metrics, output_path=None):
    """
    Create a radar chart comparing fairness metrics before and after adjustment
    
    Args:
        baseline_metrics: Dictionary with baseline fairness metrics
        fair_metrics: Dictionary with fair model fairness metrics
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Determine which metrics to plot
    metrics = [m for m in PRIMARY_FAIRNESS_METRICS if m in baseline_metrics and m in fair_metrics]
    
    if not metrics:
        raise ValueError("No common fairness metrics found in both result sets")
    
    # Setup figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Get absolute values of metrics
    baseline_values = [abs(baseline_metrics[m]) for m in metrics]
    baseline_values += baseline_values[:1]  # Close the loop
    
    fair_values = [abs(fair_metrics[m]) for m in metrics]
    fair_values += fair_values[:1]  # Close the loop
    
    # Add metric labels
    labels = [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    
    # Plot metrics
    ax.plot(angles, baseline_values, 'r-', linewidth=2, label='Baseline Model')
    ax.fill(angles, baseline_values, 'r', alpha=0.1)
    
    ax.plot(angles, fair_values, 'g-', linewidth=2, label='Fair Model')
    ax.fill(angles, fair_values, 'g', alpha=0.1)
    
    # Set y-limits
    ax.set_ylim(0, max(max(baseline_values), max(fair_values)) * 1.1)
    
    # Add labels
    plt.xticks(angles[:-1], labels)
    
    # Add title and legend
    plt.title('Fairness Metrics Comparison', fontsize=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_group_comparison_metrics(y_true, y_pred, protected_attr, output_path=None):
    """
    Plot performance metrics comparison between protected attribute groups
    
    Args:
        y_true: True class labels
        y_pred: Predicted labels
        protected_attr: Protected attribute values (0/1)
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get indices for each group
    group0_idx = (protected_attr == 0)
    group1_idx = (protected_attr == 1)
    
    # Calculate metrics for each group
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = []
    
    # Overall metrics
    metrics.append({
        'Group': 'Overall',
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    })
    
    # Group 0 metrics
    metrics.append({
        'Group': 'Group 0',
        'Accuracy': accuracy_score(y_true[group0_idx], y_pred[group0_idx]),
        'Precision': precision_score(y_true[group0_idx], y_pred[group0_idx], zero_division=0),
        'Recall': recall_score(y_true[group0_idx], y_pred[group0_idx], zero_division=0),
        'F1': f1_score(y_true[group0_idx], y_pred[group0_idx], zero_division=0)
    })
    
    # Group 1 metrics
    metrics.append({
        'Group': 'Group 1',
        'Accuracy': accuracy_score(y_true[group1_idx], y_pred[group1_idx]),
        'Precision': precision_score(y_true[group1_idx], y_pred[group1_idx], zero_division=0),
        'Recall': recall_score(y_true[group1_idx], y_pred[group1_idx], zero_division=0),
        'F1': f1_score(y_true[group1_idx], y_pred[group1_idx], zero_division=0)
    })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Plot as grouped bar chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    bar_width = 0.25
    index = np.arange(len(metric_names))
    
    # Position bars
    ax.bar(index - bar_width, metrics_df.iloc[0][metric_names], bar_width, 
           label='Overall', color='purple')
    ax.bar(index, metrics_df.iloc[1][metric_names], bar_width, 
           label='Group 0', color='blue')
    ax.bar(index + bar_width, metrics_df.iloc[2][metric_names], bar_width, 
           label='Group 1', color='red')
    
    # Add labels and formatting
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Group')
    ax.set_xticks(index)
    ax.set_xticklabels(metric_names)
    ax.legend()
    
    # Add value labels
    for i, metric in enumerate(metric_names):
        for j, group in enumerate(['Overall', 'Group 0', 'Group 1']):
            pos = i + (j-1) * bar_width
            val = metrics_df[metrics_df['Group'] == group][metric].values[0]
            ax.text(pos, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Calculate disparity
    disparity = np.abs(metrics_df.iloc[1][metric_names].values - metrics_df.iloc[2][metric_names].values)
    disparity_text = "\n".join([f"{m}: {d:.3f}" for m, d in zip(metric_names, disparity)])
    ax.text(0.02, 0.02, f"Disparity:\n{disparity_text}", transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_fairness_metrics_by_constraint(results_df, output_path=None):
    """
    Create visualizations showing how different fairness constraints affect various metrics.
    
    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save the plot (if None, just display)
        
    Returns:
        Matplotlib figure
    """
    # Check required columns
    required_metrics = ['fair_fairness_demographic_parity_difference', 
                       'fair_fairness_equal_opportunity_difference',
                       'fair_fairness_equalized_odds_difference']
    
    if not all(metric in results_df.columns for metric in required_metrics) or 'fairness_constraint' not in results_df.columns:
        raise ValueError("Results DataFrame must contain fairness metrics and fairness_constraint column")
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors and labels for constraints
    colors = {
        'demographic_parity_difference': 'blue',
        'equal_opportunity_difference': 'red',
        'equalized_odds_difference': 'green'
    }
    
    constraint_labels = {
        'demographic_parity_difference': 'Demographic Parity',
        'equal_opportunity_difference': 'Equal Opportunity',
        'equalized_odds_difference': 'Equalized Odds'
    }
    
    # Get metrics without the prefix for cleaner labels
    metric_labels = [
        'Demographic Parity Difference',
        'Equal Opportunity Difference',
        'Equalized Odds Difference'
    ]
    
    # Plot boxplots for each metric
    for i, metric in enumerate(required_metrics):
        # Prepare data for boxplot
        constraints = results_df['fairness_constraint'].unique()
        data = [results_df[results_df['fairness_constraint'] == c][metric].abs() for c in constraints]
        
        # Create boxplot
        boxplots = axes[i].boxplot(data, labels=[constraint_labels.get(c, c) for c in constraints], 
                                patch_artist=True, widths=0.6, showfliers=False)
        
        # Color the boxes
        for j, box in enumerate(boxplots['boxes']):
            box.set(facecolor=colors.get(constraints[j], 'gray'), alpha=0.6)
        
        # Add labels and formatting
        axes[i].set_ylabel('Absolute Value')
        axes[i].set_title(metric_labels[i])
        axes[i].grid(True, alpha=0.3)
        
        # Rotate x-tick labels if needed
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Effect of Fairness Constraints on Different Fairness Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig