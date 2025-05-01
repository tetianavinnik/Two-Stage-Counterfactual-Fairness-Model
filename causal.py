import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_causal_graph(causal_graph, output_path=None, figsize=(12, 10)):
    """
    Visualize the causal graph
    
    Args:
        causal_graph: CausalGraph object
        output_path: Path to save the plot (if None, just display)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig = causal_graph.visualize(figsize=figsize)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_feature_distributions(X, s, feature_names, output_path=None, n_features=9):
    """
    Plot distributions of top features by protected attribute
    
    Args:
        X: Feature matrix
        s: Protected attribute values (0/1)
        feature_names: List of feature names
        output_path: Path to save the plot (if None, just display)
        n_features: Number of features to plot
    
    Returns:
        Matplotlib figure
    """
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X
    
    X_df['protected'] = s
    
    # Calculate correlation with protected attribute
    correlations = []
    for col in X_df.columns:
        if col != 'protected':
            corr = abs(X_df[[col, 'protected']].corr().iloc[0, 1])
            correlations.append((col, corr))
    
    # Sort by correlation and select top features
    top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:n_features]
    top_feature_names = [feat[0] for feat in top_features]
    
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    # Plot distributions for each feature
    for i, feature in enumerate(top_feature_names):
        ax = axes[i]
        
        # Check if feature is categorical or continuous
        if X_df[feature].nunique() <= 5:  # Categorical
            # Create cross-tabulation
            cross_tab = pd.crosstab(
                X_df[feature], 
                X_df['protected'], 
                normalize='columns'
            ) * 100  # Convert to percentage
            
            cross_tab.plot(kind='bar', stacked=False, ax=ax)
            ax.set_xlabel(feature)
            ax.set_ylabel('Percentage')
            ax.set_title(f"{feature} (corr: {top_features[i][1]:.3f})")
            ax.legend(['Group 0', 'Group 1'])
            
        else:  # Continuous
            sns.kdeplot(x=X_df[X_df['protected'] == 0][feature], 
                     ax=ax, label='Group 0', color='blue')
            sns.kdeplot(x=X_df[X_df['protected'] == 1][feature], 
                     ax=ax, label='Group 1', color='red')
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f"{feature} (corr: {top_features[i][1]:.3f})")
            ax.legend()
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions by Protected Attribute', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_counterfactual_distributions(X, X_cf, s, feature_names, 
                                     output_path=None, n_features=9):
    """
    Plot distributions of original vs. counterfactual features
    
    Args:
        X: Original feature matrix
        X_cf: Counterfactual feature matrix
        s: Protected attribute values (0/1)
        feature_names: List of feature names
        output_path: Path to save the plot (if None, just display)
        n_features: Number of features to plot
    
    Returns:
        Matplotlib figure
    """
    # Calculate feature changes
    feature_changes = np.abs(X - X_cf).mean(axis=0)
    
    # Get indices of top changed features
    top_indices = np.argsort(-feature_changes)[:n_features]
    
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_indices):
        if feat_idx < len(feature_names):
            ax = axes[i]
            feature_name = feature_names[feat_idx]
            
            # Get original and counterfactual values
            orig_values = X[:, feat_idx]
            cf_values = X_cf[:, feat_idx]
            
            group0_mask = (s == 0)
            group1_mask = (s == 1)
            
            sns.kdeplot(x=orig_values[group0_mask], ax=ax, 
                     label='G0 Original', color='blue', alpha=0.5)
            sns.kdeplot(x=orig_values[group1_mask], ax=ax, 
                     label='G1 Original', color='red', alpha=0.5)
            
            sns.kdeplot(x=cf_values[group0_mask], ax=ax, 
                     label='G0 Counterfactual', color='blue', 
                     linestyle='--')
            sns.kdeplot(x=cf_values[group1_mask], ax=ax, 
                     label='G1 Counterfactual', color='red', 
                     linestyle='--')
            
            # Calculate mean difference
            mean_diff_g0 = np.mean(cf_values[group0_mask] - orig_values[group0_mask])
            mean_diff_g1 = np.mean(cf_values[group1_mask] - orig_values[group1_mask])
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f"{feature_name}\nΔG0={mean_diff_g0:.3f}, ΔG1={mean_diff_g1:.3f}")
            
            if i == 0:
                ax.legend(fontsize=8)
            else:
                ax.get_legend().remove()
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Original vs. Counterfactual Feature Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    fig.legend(['Group 0 Original', 'Group 1 Original', 
               'Group 0 Counterfactual', 'Group 1 Counterfactual'], 
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_embedding_space(X, X_cf, s, y=None, method='pca', output_path=None):
    """
    Visualize original and counterfactual data in a low-dimensional space
    
    Args:
        X: Original feature matrix
        X_cf: Counterfactual feature matrix
        s: Protected attribute values (0/1)
        y: Target values (optional)
        method: Dimensionality reduction method ('pca' or 'tsne')
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    # Combine original and counterfactual data
    X_combined = np.vstack([X, X_cf])

    data_type = np.array(['Original']*len(X) + ['Counterfactual']*len(X_cf))
    protected_attr = np.concatenate([s, s])
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_combined)
        title = "PCA Embedding: Original vs. Counterfactual Data"
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_combined)//10))
        embedding = reducer.fit_transform(X_combined)
        title = "t-SNE Embedding: Original vs. Counterfactual Data"
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(figsize=(12, 10))

    markers = {'Original': 'o', 'Counterfactual': '^'}
    colors = {0: 'blue', 1: 'red'}
    
    for data_t in ['Original', 'Counterfactual']:
        for group in [0, 1]:
            mask = (data_type == data_t) & (protected_attr == group)
            ax.scatter(
                embedding[mask, 0], 
                embedding[mask, 1],
                s=80 if data_t == 'Original' else 60,
                marker=markers[data_t],
                c=colors[group],
                edgecolor='black' if data_t == 'Counterfactual' else None,
                alpha=0.7 if data_t == 'Original' else 0.5,
                label=f"Group {group} ({data_t})"
            )
    
    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.set_title(title)
    ax.legend()

    ax.grid(True, alpha=0.3)
    
    annotation = (
        "Each point represents a sample:\n"
        "• Circles: Original data\n"
        "• Triangles: Counterfactual versions\n"
        "• Blue: Group 0, Red: Group 1"
    )
    ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_feature_importance(feature_importances, feature_names, 
                           output_path=None, top_n=20):
    """
    Plot feature importance
    
    Args:
        feature_importances: Array of feature importance values
        feature_names: List of feature names
        output_path: Path to save the plot (if None, just display)
        top_n: Number of top features to show
    
    Returns:
        Matplotlib figure
    """
    # Create feature importance pairs
    importance_pairs = list(zip(feature_names, feature_importances))
    
    # Sort by importance (descending)
    sorted_pairs = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
    
    # Select top N features
    top_features = sorted_pairs[:min(top_n, len(sorted_pairs))]

    names = [pair[0] for pair in top_features]
    values = [pair[1] for pair in top_features]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(np.arange(len(names)), values, color='teal')
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Display top features at the top

    ax.set_xlabel('Importance')
    ax.set_title(f'Top {len(names)} Feature Importance')

    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_feature_categories(feature_categories, feature_names, 
                           output_path=None):
    """
    Visualize feature categorization (direct, proxy, mediator, neutral)
    
    Args:
        feature_categories: Dictionary with feature category lists
        feature_names: List of all feature names
        output_path: Path to save the plot (if None, just display)
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    category_colors = {
        'direct': 'gold',
        'proxy': 'lightcoral',
        'mediator': 'lightseagreen', 
        'neutral': 'lightskyblue'
    }
    
    # Count features in each category
    category_counts = {cat: len(indices) for cat, indices in feature_categories.items()}

    categories = list(category_counts.keys())
    counts = [category_counts[cat] for cat in categories]

    bars = ax.bar(categories, counts, color=[category_colors[cat] for cat in categories])

    ax.set_xlabel('Feature Category')
    ax.set_ylabel('Number of Features')
    ax.set_title('Feature Categorization')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                f'{int(height)}', ha='center', va='bottom')

    descriptions = {
        'direct': 'Directly affected by\nprotected attribute',
        'proxy': 'Can serve as proxies for\nprotected attribute',
        'mediator': 'Mediate between protected\nattribute and outcome',
        'neutral': 'Not related to\nprotected attribute'
    }

    for i, cat in enumerate(categories):
        ax.text(i, counts[i] + 1, descriptions[cat], ha='center', va='bottom', 
                fontsize=9, color='dimgray')
    
    # List specific features in each category (if not too many)
    if sum(counts) < 50:  # Only if total number of features is manageable
        feature_lists = {}
        for cat, indices in feature_categories.items():
            if indices:
                feature_lists[cat] = [feature_names[i] for i in indices]

        text = ""
        for cat in categories:
            if cat in feature_lists:
                text += f"{cat.capitalize()} features:\n"
                text += ", ".join(feature_lists[cat][:10])  # Show up to 10 features
                if len(feature_lists[cat]) > 10:
                    text += f", ... (+{len(feature_lists[cat])-10} more)"
                text += "\n\n"
        
        if text:
            plt.figtext(0.5, 0.01, text, ha="center", fontsize=9, 
                       bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout(rect=[0, 0.1 if sum(counts) < 50 else 0, 1, 1])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig
