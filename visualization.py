"""
Visualization utilities for the Two-Stage Counterfactual Fairness Model (TSCFM).
This module provides visualization functions to analyze causal relationships,
counterfactual effects, and fairness metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from fairness_metrics import (
    roc_curve, auc
)


def plot_feature_distributions(X: pd.DataFrame, 
                             s: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             n_cols: int = 3,
                             figsize: Tuple[int, int] = None,
                             group_names: List[str] = ['Male', 'Female'],
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distributions of features stratified by protected attribute.
    
    Args:
        X: Feature matrix
        s: Protected attribute values
        feature_names: Names of features (if None, use column names from X if available)
        n_cols: Number of columns in the grid
        figsize: Size of the figure
        group_names: Names of the groups (e.g., ['Male', 'Female'])
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure
    """
    # If X is a DataFrame, use column names
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_values = X.values
    else:
        X_values = X
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    # Determine number of features to plot
    n_features = min(16, X_values.shape[1])

    n_rows = int(np.ceil(n_features / n_cols))
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    plot_df = pd.DataFrame(X_values[:, :n_features], columns=feature_names[:n_features])
    plot_df['group'] = [group_names[s_val] for s_val in s]

    for i, feature in enumerate(feature_names[:n_features]):
        ax = axes[i]
        
        if plot_df[feature].nunique() <= 5:  # Categorical/binary feature
            # Use countplot for categorical features
            sns.countplot(x=feature, hue='group', data=plot_df, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            ax.set_ylabel("Count")
            if i % n_cols == 0:
                ax.set_ylabel("Count")
            else:
                ax.set_ylabel("")
        else:  # Continuous feature
            # Use KDE plot for continuous features
            sns.kdeplot(x=feature, hue='group', data=plot_df, fill=True, common_norm=False, alpha=0.6, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            if i % n_cols == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")
        
        # Rotate x-axis labels if there are many unique values
        if plot_df[feature].nunique() > 10:
            ax.tick_params(axis='x', rotation=45)
        
        # Only show legend for the first subplot
        if i > 0:
            ax.get_legend().remove()
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Feature Distributions by Protected Group", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_causal_effects(causal_graph, 
                       feature_names: List[str],
                       coefficients: Dict[str, float],
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize causal effects in the graph with edge weights.
    
    Args:
        causal_graph: CausalGraph object
        feature_names: Names of features
        coefficients: Dictionary mapping (cause, effect) pairs to effect sizes
        figsize: Size of the figure
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the graph from causal_graph
    graph = causal_graph.graph

    layout = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)

    node_colors = []
    for node in graph.nodes():
        if node == causal_graph.protected_attribute:
            node_colors.append('orangered')
        elif node in causal_graph.direct_features:
            node_colors.append('gold')
        elif node in causal_graph.proxy_features:
            node_colors.append('lightcoral')
        elif node in causal_graph.mediator_features:
            node_colors.append('lightseagreen')
        elif node in causal_graph.neutral_features:
            node_colors.append('lightskyblue')
        else:
            node_colors.append('mediumpurple')
 
    nx.draw_networkx_nodes(
        graph, 
        layout, 
        node_color=node_colors,
        node_size=1000,
        alpha=0.8,
        ax=ax
    )

    nx.draw_networkx_labels(
        graph, 
        layout, 
        font_size=10,
        font_weight='bold',
        ax=ax
    )

    for edge in graph.edges():
        effect_size = abs(coefficients.get(edge, 0.1))
        # Scale effect size for visualization (adjust as needed)
        width = max(1, min(8, effect_size * 10))

        color = 'blue' if coefficients.get(edge, 0) >= 0 else 'red'
        
        nx.draw_networkx_edges(
            graph,
            layout,
            edgelist=[edge],
            width=width,
            edge_color=color,
            connectionstyle='arc3,rad=0.1',
            arrowstyle='-|>',
            arrowsize=20,
            alpha=0.7,
            ax=ax
        )
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orangered', markersize=10, label='Protected Attribute'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, label='Direct Effect'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Proxy'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightseagreen', markersize=10, label='Mediator'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightskyblue', markersize=10, label='Neutral'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumpurple', markersize=10, label='Outcome'),
    ]

    legend_elements.extend([
        plt.Line2D([0], [0], color='blue', lw=4, label='Positive Effect'),
        plt.Line2D([0], [0], color='red', lw=4, label='Negative Effect')
    ])
    
    ax.legend(handles=legend_elements, loc='upper right')

    plt.title('Causal Effect Visualization', fontsize=16)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_counterfactual_distributions(X: np.ndarray,
                                    X_cf: np.ndarray,
                                    s: np.ndarray,
                                    feature_names: List[str],
                                    feature_indices: Optional[List[int]] = None,
                                    n_cols: int = 3,
                                    figsize: Optional[Tuple[int, int]] = None,
                                    save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot original vs. counterfactual feature distributions for a subset of features.
    
    Args:
        X: Original feature matrix
        X_cf: Counterfactual feature matrix
        s: Protected attribute values
        feature_names: Names of all features
        feature_indices: Indices of features to plot (if None, select features with largest changes)
        n_cols: Number of columns in the grid
        figsize: Size of the figure
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure or None
    """
    try:
        # If feature_indices not provided, select features with largest changes
        if feature_indices is None:
            # Calculate mean absolute difference for each feature
            mean_abs_diff = np.mean(np.abs(X - X_cf), axis=0)
            # Get indices of top 9 features with largest changes
            feature_indices = np.argsort(-mean_abs_diff)[:9]
 
        n_features = len(feature_indices)

        n_rows = int(np.ceil(n_features / n_cols))
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
  
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Convert to list if there's only one axis
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, feat_idx in enumerate(feature_indices):
            if i >= len(axes):
                break  # Safety check to avoid index errors
                
            ax = axes[i]
            feature_name = feature_names[feat_idx]
            
            # Get original values by gender
            x_male = X[s == 0, feat_idx]
            x_female = X[s == 1, feat_idx]
            
            # Get counterfactual values by gender
            x_cf_male = X_cf[s == 0, feat_idx]
            x_cf_female = X_cf[s == 1, feat_idx]
 
            unique_vals = np.unique(np.concatenate([x_male, x_female]))
            if len(unique_vals) <= 5:  # Categorical/binary feature
                # Create dataframe for plotting
                df = pd.DataFrame({
                    'Value': np.concatenate([x_male, x_female, x_cf_male, x_cf_female]),
                    'Type': ['Original']*len(x_male) + ['Original']*len(x_female) + 
                        ['Counterfactual']*len(x_cf_male) + ['Counterfactual']*len(x_cf_female),
                    'Gender': ['Male']*len(x_male) + ['Female']*len(x_female) + 
                            ['Male']*len(x_cf_male) + ['Female']*len(x_cf_female)
                })

                sns.countplot(x='Value', hue='Type', data=df, ax=ax)
                ax.set_title(f"Distribution of {feature_name}")
            else: 
                # Plot original densities
                sns.kdeplot(x=x_male, ax=ax, label='Male (Original)', color='blue', fill=True, alpha=0.3)
                sns.kdeplot(x=x_female, ax=ax, label='Female (Original)', color='red', fill=True, alpha=0.3)
                
                # Plot counterfactual densities
                sns.kdeplot(x=x_cf_male, ax=ax, label='Male (Counterfactual)', color='blue', linestyle='--')
                sns.kdeplot(x=x_cf_female, ax=ax, label='Female (Counterfactual)', color='red', linestyle='--')
                
                ax.set_title(f"Distribution of {feature_name}")

            # Fix: Check if legend exists before trying to remove it
            if i > 0:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
  
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle("Original vs. Counterfactual Feature Distributions", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    except Exception as e:
        import logging
        logger = logging.getLogger("counterfactual_generator")
        logger.error(f"Error in visualization: {e}")
        return None

def plot_embedding_space(X: np.ndarray, 
                        X_cf: np.ndarray,
                        s: np.ndarray,
                        y: Optional[np.ndarray] = None,
                        method: str = 'pca',
                        figsize: Tuple[int, int] = (12, 10),
                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualize original and counterfactual data points in a low-dimensional embedding space.
    
    Args:
        X: Original feature matrix
        X_cf: Counterfactual feature matrix
        s: Protected attribute values (gender)
        y: Target values (optional)
        method: Dimensionality reduction method ('pca' or 'tsne')
        figsize: Size of the figure
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure or None
    """
    try:
        # Combine original and counterfactual data
        X_combined = np.vstack([X, X_cf])
        
        # Add a column to track original vs. counterfactual
        data_type = np.array(['Original'] * len(X) + ['Counterfactual'] * len(X_cf))
        
        # Add gender labels for both sets
        gender_labels = np.concatenate([s, s])
        gender_names = np.array(['Male' if g == 0 else 'Female' for g in gender_labels])
        
        # Add outcome labels if provided
        if y is not None:
            outcome_labels = np.concatenate([y, y])
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_combined)
            title = "PCA Embedding of Original vs. Counterfactual Data"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedding = reducer.fit_transform(X_combined)
            title = "t-SNE Embedding of Original vs. Counterfactual Data"
        else:
            raise ValueError(f"Unknown method: {method}")
 
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with separate styles for original vs. counterfactual
        for data_t in ['Original', 'Counterfactual']:
            for gender in ['Male', 'Female']:
                mask = (data_type == data_t) & (gender_names == gender)

                marker = 'o' if data_t == 'Original' else '^'

                color = 'blue' if gender == 'Male' else 'red'
   
                alpha = 0.7 if data_t == 'Original' else 0.5
                edgecolor = 'black' if data_t == 'Counterfactual' else None
                
                label = f"{gender} ({data_t})"

                ax.scatter(
                    embedding[mask, 0], 
                    embedding[mask, 1], 
                    c=color, 
                    marker=marker, 
                    s=100, 
                    alpha=alpha,
                    edgecolor=edgecolor,
                    label=label
                )

        if y is not None:
            # Plot decision boundary by coloring the background
            x_min, x_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
            y_min, y_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            
            # Use a simple KNN classifier to approximate the decision boundary
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=15)
            clf.fit(embedding, outcome_labels)
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
 
            contour = ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
            plt.colorbar(contour, ax=ax, label='Probability of Positive Outcome')

        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
 
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
        ax.grid(True, alpha=0.3)
 
        annotation_text = (
            "Each point represents a data sample.\n"
            "Circles: Original data\n"
            "Triangles: Counterfactual versions\n"
            "Blue: Male, Red: Female"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.02, annotation_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    except Exception as e:
        import logging
        logger = logging.getLogger("counterfactual_generator")
        logger.error(f"Error in visualization: {e}")
        return None

def plot_feature_distributions(X: pd.DataFrame, 
                             s: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             n_cols: int = 3,
                             figsize: Tuple[int, int] = None,
                             group_names: List[str] = ['Male', 'Female'],
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot distributions of features stratified by protected attribute.
    
    Args:
        X: Feature matrix
        s: Protected attribute values
        feature_names: Names of features (if None, use column names from X if available)
        n_cols: Number of columns in the grid
        figsize: Size of the figure
        group_names: Names of the groups (e.g., ['Male', 'Female'])
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure or None
    """
    try:
        # If X is a DataFrame, use column names
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
        n_features = min(16, X_values.shape[1])  # Limit to 16 features

        n_rows = int(np.ceil(n_features / n_cols))
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        plot_df = pd.DataFrame(X_values[:, :n_features], columns=feature_names[:n_features])
        plot_df['group'] = [group_names[s_val] for s_val in s]

        for i, feature in enumerate(feature_names[:n_features]):
            if i >= len(axes):
                break  # Safety check
                
            ax = axes[i]
            
            if plot_df[feature].nunique() <= 5:  # Categorical/binary feature
                # Use countplot for categorical features
                sns.countplot(x=feature, hue='group', data=plot_df, ax=ax)
                ax.set_title(f"Distribution of {feature}")
                ax.set_ylabel("Count")
                if i % n_cols == 0:
                    ax.set_ylabel("Count")
                else:
                    ax.set_ylabel("")
            else:  # Continuous feature
                # Use KDE plot for continuous features
                sns.kdeplot(x=feature, hue='group', data=plot_df, fill=True, common_norm=False, alpha=0.6, ax=ax)
                ax.set_title(f"Distribution of {feature}")
                if i % n_cols == 0:
                    ax.set_ylabel("Density")
                else:
                    ax.set_ylabel("")
 
            if plot_df[feature].nunique() > 10:
                ax.tick_params(axis='x', rotation=45)

            if i > 0:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle("Feature Distributions by Protected Group", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    except Exception as e:
        import logging
        logger = logging.getLogger("counterfactual_generator")
        logger.error(f"Error in visualization: {e}")
        return None

def plot_fairness_metrics_comparison(metrics_before: Dict[str, float],
                                   metrics_after: Dict[str, float],
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Compare fairness metrics before and after counterfactual adjustments.
    
    Args:
        metrics_before: Dictionary of fairness metrics before adjustment
        metrics_after: Dictionary of fairness metrics after adjustment
        figsize: Size of the figure
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure or None
    """
    try:
        valid_metrics = []
        for metric in metrics_before:
            if (metrics_before[metric] != float('inf') and not np.isnan(metrics_before[metric]) and
                metrics_after[metric] != float('inf') and not np.isnan(metrics_after[metric])):
                valid_metrics.append(metric)
 
        if not valid_metrics:
            return None

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(valid_metrics))
        width = 0.35
        
        before_bars = ax.bar(x - width/2, [metrics_before[m] for m in valid_metrics], width, 
                            label='Before Adjustment', color='indianred')
        after_bars = ax.bar(x + width/2, [metrics_after[m] for m in valid_metrics], width,
                        label='After Adjustment', color='seagreen')

        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Fairness Metrics Before and After Counterfactual Adjustment', fontsize=14)
        ax.set_xticks(x)

        metric_labels = [m.replace('_', ' ').title() for m in valid_metrics]
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
 
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        add_labels(before_bars)
        add_labels(after_bars)
        

        ax.legend()

        ax.grid(True, axis='y', alpha=0.3)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()

        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    except Exception as e:
        import logging
        logger = logging.getLogger("TSCFM")
        logger.error(f"Error in visualizations: {e}")
        return None

def plot_outcome_probabilities(y_prob_before: np.ndarray, 
                             y_prob_after: np.ndarray,
                             s: np.ndarray,
                             n_bins: int = 10,
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot histograms of outcome probabilities before and after counterfactual adjustments,
    stratified by protected attribute.
    
    Args:
        y_prob_before: Outcome probabilities before adjustment
        y_prob_after: Outcome probabilities after adjustment
        s: Protected attribute values
        n_bins: Number of bins for histograms
        figsize: Size of the figure
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure or None
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Split data by gender
        male_mask = (s == 0)
        female_mask = (s == 1)
        
        # Plot histograms for before adjustment
        ax1.hist(y_prob_before[male_mask], bins=n_bins, alpha=0.7, label='Male', color='blue')
        ax1.hist(y_prob_before[female_mask], bins=n_bins, alpha=0.7, label='Female', color='red')
        
        # Calculate and show means
        male_mean_before = np.mean(y_prob_before[male_mask])
        female_mean_before = np.mean(y_prob_before[female_mask])
        
        ax1.axvline(male_mean_before, color='blue', linestyle='--', linewidth=2)
        ax1.axvline(female_mean_before, color='red', linestyle='--', linewidth=2)
        
        # Calculate probability shift (demographic parity difference)
        dp_diff_before = abs(male_mean_before - female_mean_before)

        ax1.text(0.05, 0.95, f"Male Mean: {male_mean_before:.3f}", transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', color='blue')
        ax1.text(0.05, 0.90, f"Female Mean: {female_mean_before:.3f}", transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', color='red')
        ax1.text(0.05, 0.85, f"DP Diff: {dp_diff_before:.3f}", transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold')

        ax1.set_xlabel('Probability of Positive Outcome', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Before Counterfactual Adjustment', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(y_prob_after[male_mask], bins=n_bins, alpha=0.7, label='Male', color='blue')
        ax2.hist(y_prob_after[female_mask], bins=n_bins, alpha=0.7, label='Female', color='red')
        
        # Calculate and show means
        male_mean_after = np.mean(y_prob_after[male_mask])
        female_mean_after = np.mean(y_prob_after[female_mask])
        
        ax2.axvline(male_mean_after, color='blue', linestyle='--', linewidth=2)
        ax2.axvline(female_mean_after, color='red', linestyle='--', linewidth=2)
        
        # Calculate probability shift (demographic parity difference)
        dp_diff_after = abs(male_mean_after - female_mean_after)

        ax2.text(0.05, 0.95, f"Male Mean: {male_mean_after:.3f}", transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', color='blue')
        ax2.text(0.05, 0.90, f"Female Mean: {female_mean_after:.3f}", transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', color='red')
        ax2.text(0.05, 0.85, f"DP Diff: {dp_diff_after:.3f}", transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold')
        

        if dp_diff_before > 0:  # Avoid division by zero
            improvement = (dp_diff_before - dp_diff_after) / dp_diff_before * 100
            ax2.text(0.05, 0.80, f"Improvement: {improvement:.1f}%", transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', color='green' if improvement > 0 else 'red', fontweight='bold')

        ax2.set_xlabel('Probability of Positive Outcome', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('After Counterfactual Adjustment', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        

        plt.suptitle('Distribution of Outcome Probabilities by Gender', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.show()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    except Exception as e:
        import logging
        logger = logging.getLogger("TSCFM")
        logger.error(f"Error in visualizations: {e}")
        return None


def plot_roc_curves_combined(y_true: np.ndarray, 
                           baseline_probs: np.ndarray, 
                           fair_probs: np.ndarray, 
                           s: np.ndarray,
                           group_names: List[str] = ['Group 0', 'Group 1'],
                           title: str = 'ROC Curves Comparison',
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot ROC curves comparing baseline and fair models for different groups.
    
    Args:
        y_true: True labels
        baseline_probs: Predicted probabilities from baseline model
        fair_probs: Predicted probabilities from fair model
        s: Protected attribute values
        group_names: Names of the groups
        title: Title of the plot
        figsize: Size of the figure
        save_path: If provided, save the plot to this path
        
    Returns:
        Matplotlib figure object or None if an error occurs
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['blue', 'red']
        linestyles = ['-', '--']
        
        # Calculate and plot ROC curve for each group and model
        for i, group_name in enumerate(group_names):
            mask = (s == i)
            if np.sum(mask) == 0 or len(np.unique(y_true[mask])) < 2:
                continue
            
            # Baseline model
            fpr_base, tpr_base, _ = roc_curve(y_true[mask], baseline_probs[mask])
            roc_auc_base = auc(fpr_base, tpr_base)
            ax.plot(fpr_base, tpr_base, color=colors[i], linestyle=linestyles[0], lw=2,
                   label=f'{group_name} (Baseline, AUC={roc_auc_base:.3f})')
            
            # Fair model
            fpr_fair, tpr_fair, _ = roc_curve(y_true[mask], fair_probs[mask])
            roc_auc_fair = auc(fpr_fair, tpr_fair)
            ax.plot(fpr_fair, tpr_fair, color=colors[i], linestyle=linestyles[1], lw=2,
                   label=f'{group_name} (Fair, AUC={roc_auc_fair:.3f})')
            
            # Calculate ABROCA between baseline and fair for this group
            if np.array_equal(fpr_base, fpr_fair):
                abroca_group = np.abs(tpr_base - tpr_fair).mean()
            else:
                # Interpolate to common FPR points
                common_fpr = np.sort(np.unique(np.concatenate([fpr_base, fpr_fair])))
                tpr_base_interp = np.interp(common_fpr, fpr_base, tpr_base)
                tpr_fair_interp = np.interp(common_fpr, fpr_fair, tpr_fair)
                abroca_group = np.trapz(np.abs(tpr_base_interp - tpr_fair_interp), common_fpr)
            
            # Add group-specific ABROCA annotation
            ax.text(0.7, 0.1 + i*0.06, f'ABROCA {group_name}: {abroca_group:.4f}', 
                   transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        

        annotation_text = (
            "Solid lines: Baseline model\n"
            "Dashed lines: Fair model\n"
            "Blue: Group 0, Red: Group 1\n"
            "ABROCA: Area between ROC curves"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.05, annotation_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    except Exception as e:
        import logging
        logger = logging.getLogger("visualization")
        logger.error(f"Error in plot_roc_curves_combined: {e}")
        return None


def plot_feature_importance(feature_importances: np.ndarray,
                          feature_names: List[str],
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importances.
    
    Args:
        feature_importances: Array of feature importance values
        feature_names: Names of features
        top_n: Number of top features to display
        figsize: Size of the figure
        save_path: If provided, save the visualization to this path
        
    Returns:
        Matplotlib figure
    """
    df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    
    # Sort by importance and select top N
    df = df.sort_values('Importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(np.arange(len(df)), df['Importance'], color='teal')

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['Feature'])
    

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                ha='left', va='center', fontsize=9)

    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

