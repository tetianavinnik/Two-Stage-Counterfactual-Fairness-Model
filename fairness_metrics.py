"""
Fairness metrics for evaluating the Two-Stage
Counterfactual Fairness Model (TSCFM).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import integrate


def demographic_parity_difference(y_pred: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the demographic parity difference.
    
    Demographic parity requires the prediction to be independent of the protected attribute:
    P(Ŷ=1|S=0) = P(Ŷ=1|S=1)
    
    Args:
        y_pred: Predicted labels
        s: Protected attribute values (binary)
        
    Returns:
        Absolute difference in positive prediction rates between groups
    """
    # Calculate positive prediction rate for each group
    positive_rate_0 = y_pred[s == 0].mean()
    positive_rate_1 = y_pred[s == 1].mean()
    
    # Calculate the absolute difference
    dpd = np.abs(positive_rate_0 - positive_rate_1)
    
    return dpd


def disparate_impact_ratio(y_pred: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the disparate impact ratio.
    
    Disparate impact is the ratio of positive prediction rates between groups:
    DI = P(Ŷ=1|S=1) / P(Ŷ=1|S=0)
    
    Values close to 1 indicate fairness. Values below 0.8 typically indicate disparate impact.
    
    Args:
        y_pred: Predicted labels
        s: Protected attribute values (binary)
        
    Returns:
        Ratio of positive prediction rates between groups
    """
    # Calculate positive prediction rate for each group
    positive_rate_0 = y_pred[s == 0].mean()
    positive_rate_1 = y_pred[s == 1].mean()

    if positive_rate_0 == 0:
        return float('inf')
    
    # Calculate the ratio
    di = positive_rate_1 / positive_rate_0
    
    return di


def equalized_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the equalized odds difference.
    
    Equalized odds requires equal true positive rates and false positive rates across groups:
    P(Ŷ=1|Y=y,S=0) = P(Ŷ=1|Y=y,S=1) for y in {0,1}
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        s: Protected attribute values (binary)
        
    Returns:
        Sum of absolute differences in TPR and FPR between groups
    """
    # Calculate true positive rates for each group
    tpr_0 = (y_pred[(y_true == 1) & (s == 0)] == 1).mean() if np.any((y_true == 1) & (s == 0)) else 0
    tpr_1 = (y_pred[(y_true == 1) & (s == 1)] == 1).mean() if np.any((y_true == 1) & (s == 1)) else 0
    
    # Calculate false positive rates for each group
    fpr_0 = (y_pred[(y_true == 0) & (s == 0)] == 1).mean() if np.any((y_true == 0) & (s == 0)) else 0
    fpr_1 = (y_pred[(y_true == 0) & (s == 1)] == 1).mean() if np.any((y_true == 0) & (s == 1)) else 0
    
    # Calculate the difference in true positive rates and false positive rates
    tpr_diff = np.abs(tpr_0 - tpr_1)
    fpr_diff = np.abs(fpr_0 - fpr_1)
    
    # Calculate the sum of differences
    eod = tpr_diff + fpr_diff
    
    return eod


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the equal opportunity difference.
    
    Equal opportunity requires equal true positive rates across groups:
    P(Ŷ=1|Y=1,S=0) = P(Ŷ=1|Y=1,S=1)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        s: Protected attribute values (binary)
        
    Returns:
        Absolute difference in true positive rates between groups
    """
    # Calculate true positive rates for each group
    tpr_0 = (y_pred[(y_true == 1) & (s == 0)] == 1).mean() if np.any((y_true == 1) & (s == 0)) else 0
    tpr_1 = (y_pred[(y_true == 1) & (s == 1)] == 1).mean() if np.any((y_true == 1) & (s == 1)) else 0
    
    # Calculate the absolute difference in true positive rates
    eod = np.abs(tpr_0 - tpr_1)
    
    return eod


def predictive_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the predictive parity difference.
    
    Predictive parity requires equal positive predictive values across groups:
    P(Y=1|Ŷ=1,S=0) = P(Y=1|Ŷ=1,S=1)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        s: Protected attribute values (binary)
        
    Returns:
        Absolute difference in positive predictive values between groups
    """
    # Calculate positive predictive values for each group
    ppv_0 = (y_true[(y_pred == 1) & (s == 0)] == 1).mean() if np.any((y_pred == 1) & (s == 0)) else 0
    ppv_1 = (y_true[(y_pred == 1) & (s == 1)] == 1).mean() if np.any((y_pred == 1) & (s == 1)) else 0
    
    # Calculate the absolute difference in positive predictive values
    ppd = np.abs(ppv_0 - ppv_1)
    
    return ppd


def predictive_equality_difference(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the predictive equality difference.
    
    Predictive equality requires equal false positive rates across groups:
    P(Ŷ=1|Y=0,S=0) = P(Ŷ=1|Y=0,S=1)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        s: Protected attribute values (binary)
        
    Returns:
        Absolute difference in false positive rates between groups
    """
    # Calculate false positive rates for each group
    fpr_0 = (y_pred[(y_true == 0) & (s == 0)] == 1).mean() if np.any((y_true == 0) & (s == 0)) else 0
    fpr_1 = (y_pred[(y_true == 0) & (s == 1)] == 1).mean() if np.any((y_true == 0) & (s == 1)) else 0
    
    # Calculate the absolute difference in false positive rates
    ped = np.abs(fpr_0 - fpr_1)
    
    return ped


def treatment_equality_difference(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the treatment equality difference.
    
    Treatment equality requires equal ratios of false negatives to false positives across groups:
    (FN/FP)_group0 = (FN/FP)_group1
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        s: Protected attribute values (binary)
        
    Returns:
        Absolute difference in FN/FP ratios between groups
    """
    # Calculate confusion matrix elements for each group
    cm_0 = confusion_matrix(y_true[s == 0], y_pred[s == 0])
    cm_1 = confusion_matrix(y_true[s == 1], y_pred[s == 1])
    
    # Extract false negatives and false positives
    if cm_0.shape[0] > 1 and cm_0.shape[1] > 1:
        fn_0, fp_0 = cm_0[1, 0], cm_0[0, 1]
    else:
        # Handle case where confusion matrix doesn't have all classes
        fn_0, fp_0 = 0, 0
    
    if cm_1.shape[0] > 1 and cm_1.shape[1] > 1:
        fn_1, fp_1 = cm_1[1, 0], cm_1[0, 1]
    else:
        # Handle case where confusion matrix doesn't have all classes
        fn_1, fp_1 = 0, 0
    
    # Calculate FN/FP ratios (avoid division by zero)
    ratio_0 = fn_0 / fp_0 if fp_0 > 0 else float('inf')
    ratio_1 = fn_1 / fp_1 if fp_1 > 0 else float('inf')
    
    # If both ratios are infinite, they're equal
    if ratio_0 == float('inf') and ratio_1 == float('inf'):
        return 0
    
    # If only one ratio is infinite, they're maximally different
    if ratio_0 == float('inf') or ratio_1 == float('inf'):
        return float('inf')
    
    # Calculate the absolute difference in ratios
    ted = np.abs(ratio_0 - ratio_1)
    
    return ted


def abroca_score(y_true: np.ndarray, y_score: np.ndarray, s: np.ndarray) -> float:
    """
    Calculate the ABROCA (Area Between ROC Curves) score.
    
    ABROCA measures the area between the ROC curves of different groups:
    ABROCA = ∫|ROC_curve(group0) - ROC_curve(group1)| dx
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        s: Protected attribute values (binary)
        
    Returns:
        Area between ROC curves
    """
    # Calculate ROC curve for each group
    fpr_0, tpr_0, _ = roc_curve(y_true[s == 0], y_score[s == 0])
    fpr_1, tpr_1, _ = roc_curve(y_true[s == 1], y_score[s == 1])
    
    # Interpolate ROC curves to the same set of FPR points
    combined_fpr = np.sort(np.unique(np.concatenate([fpr_0, fpr_1])))
    
    # Interpolate TPR values for both curves
    tpr_0_interp = np.interp(combined_fpr, fpr_0, tpr_0)
    tpr_1_interp = np.interp(combined_fpr, fpr_1, tpr_1)
    
    # Calculate area between curves
    abroca = np.trapz(np.abs(tpr_0_interp - tpr_1_interp), combined_fpr)
    
    return abroca


def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray, s: np.ndarray,
                   group_names: List[str] = ['Group 0', 'Group 1'],
                   title: str = 'ROC Curves by Group',
                   save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot ROC curves for different groups.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        s: Protected attribute values (binary)
        group_names: Names of the groups
        title: Title of the plot
        save_path: If provided, save the plot to this path
        
    Returns:
        Figure object or None if an error occurs
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate ROC curve for each group
        auc_values = []
        for group_idx, group_name in enumerate(group_names):
            mask = (s == group_idx)
            if np.sum(mask) == 0 or len(np.unique(y_true[mask])) < 2:
                continue
                
            fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
            roc_auc = auc(fpr, tpr)
            auc_values.append(roc_auc)
            
            ax.plot(fpr, tpr, lw=2,
                    label=f'{group_name} (AUC = {roc_auc:.3f})')
        
        # Calculate ABROCA if we have both groups
        if len(auc_values) == 2:
            try:
                abroca = abroca_score(y_true, y_score, s)
                
                # Add ABROCA to the plot
                ax.text(0.6, 0.1, f'ABROCA = {abroca:.4f}', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
            except Exception as e:
                # If ABROCA calculation fails, just continue without it
                pass
        
        # Add reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        
        # Set plot details
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    except Exception as e:
        import logging
        logger = logging.getLogger("fairness_metrics")
        logger.error(f"Error in plot_roc_curves: {e}")
        return None


def fairness_report(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, 
                   s: np.ndarray, group_names: List[str] = ['Group 0', 'Group 1']) -> Dict[str, Dict[str, float]]:
    """
    Generate a comprehensive fairness report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Predicted probabilities (if available, otherwise pass y_pred)
        s: Protected attribute values (binary)
        group_names: Names of the groups
        
    Returns:
        Dictionary with fairness metrics
    """
    # Performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_true, y_score)
    except:
        auc_score = np.nan
    
    # Fairness metrics
    dpd = demographic_parity_difference(y_pred, s)
    dir_value = disparate_impact_ratio(y_pred, s)
    eod = equalized_odds_difference(y_true, y_pred, s)
    eopd = equal_opportunity_difference(y_true, y_pred, s)
    ppd = predictive_parity_difference(y_true, y_pred, s)
    ped = predictive_equality_difference(y_true, y_pred, s)
    
    try:
        ted = treatment_equality_difference(y_true, y_pred, s)
    except:
        ted = np.nan
    
    try:
        abroca = abroca_score(y_true, y_score, s)
    except:
        abroca = np.nan
    
    # Group-specific metrics
    metrics_by_group = {}
    for group_idx, group_name in enumerate(group_names):
        mask = (s == group_idx)
        if np.sum(mask) == 0:
            continue
            
        group_accuracy = accuracy_score(y_true[mask], y_pred[mask])
        group_precision = precision_score(y_true[mask], y_pred[mask], zero_division=0)
        group_recall = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        group_f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
        
        try:
            group_auc = roc_auc_score(y_true[mask], y_score[mask])
        except:
            group_auc = np.nan
        
        metrics_by_group[group_name] = {
            'accuracy': group_accuracy,
            'precision': group_precision,
            'recall': group_recall,
            'f1': group_f1,
            'auc': group_auc,
            'positive_rate': y_pred[mask].mean()
        }
    
    # Combine all metrics
    report = {
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score
        },
        'fairness': {
            'demographic_parity_difference': dpd,
            'disparate_impact_ratio': dir_value,
            'equalized_odds_difference': eod,
            'equal_opportunity_difference': eopd,
            'predictive_parity_difference': ppd,
            'predictive_equality_difference': ped,
            'treatment_equality_difference': ted,
            'abroca': abroca
        },
        'by_group': metrics_by_group
    }
    
    return report


def print_fairness_report(report: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted fairness report.
    
    Args:
        report: The fairness report generated by fairness_report()
    """
    print("="*80)
    print("FAIRNESS REPORT")
    print("="*80)
    
    # Overall performance metrics
    print("\nOVERALL PERFORMANCE:")
    print("-"*80)
    overall = report['overall']
    for metric, value in overall.items():
        print(f"{metric.upper():20s}: {value:.4f}")
    
    # Fairness metrics
    print("\nFAIRNESS METRICS:")
    print("-"*80)
    fairness = report['fairness']
    for metric, value in fairness.items():
        if value == float('inf') or np.isnan(value):
            print(f"{metric.upper():30s}: {'N/A':>10s}")
        else:
            print(f"{metric.upper():30s}: {value:>10.4f}")
    
    # Group-specific metrics
    print("\nPERFORMANCE BY GROUP:")
    print("-"*80)
    by_group = report['by_group']
    
    # Calculate the width needed for the group name column
    max_group_name_len = max(len(group) for group in by_group.keys())
    col_width = max(20, max_group_name_len + 2)
    
    # Print header
    header = f"{'GROUP':{col_width}s}"
    metrics = list(next(iter(by_group.values())).keys())
    for metric in metrics:
        header += f"{metric.upper():>10s}"
    print(header)
    print("-"*80)
    
    # Print metrics for each group
    for group_name, group_metrics in by_group.items():
        row = f"{group_name:{col_width}s}"
        for metric, value in group_metrics.items():
            row += f"{value:>10.4f}"
        print(row)
    
    print("="*80)
