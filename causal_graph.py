"""
Causal graph implementation for the Two-Stage Counterfactual Fairness Model (TSCFM).
This module provides tools for discovering, representing, and manipulating causal structures
between features, particularly focusing on the influence of gender on credit decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union
import networkx as nx
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from itertools import combinations


class CausalGraph:
    """
    Represents and manipulates the causal structure for counterfactual fairness.
    """
    
    def __init__(self, 
                 feature_names: List[str] = None,
                 direct_effects: List[Tuple[str, str]] = None,
                 protected_attribute: str = "gender"):
        """
        Initialize a causal graph.
        
        Args:
            feature_names: Names of features in the dataset
            direct_effects: List of (cause, effect) tuples indicating direct causal relationships
            protected_attribute: Name of the protected attribute (default: "gender")
        """
        self.graph = nx.DiGraph()
        self.protected_attribute = protected_attribute
        
        # Add nodes if provided
        if feature_names:
            self.graph.add_nodes_from(feature_names)
        
        # Add edges if provided
        if direct_effects:
            self.graph.add_edges_from(direct_effects)
        
        # Categorized features (to be populated)
        self.direct_features: Set[str] = set()
        self.proxy_features: Set[str] = set()
        self.mediator_features: Set[str] = set()
        self.neutral_features: Set[str] = set()
    
    def add_node(self, node_name: str) -> None:
        """
        Add a node to the causal graph.
        
        Args:
            node_name: Name of the node to add
        """
        self.graph.add_node(node_name)
    
    def add_edge(self, cause: str, effect: str) -> None:
        """
        Add a directed edge (causal relationship) to the graph.
        
        Args:
            cause: Source node (cause)
            effect: Target node (effect)
        """
        self.graph.add_edge(cause, effect)
    
    def remove_edge(self, cause: str, effect: str) -> None:
        """
        Remove a directed edge from the graph.
        
        Args:
            cause: Source node (cause)
            effect: Target node (effect)
        """
        self.graph.remove_edge(cause, effect)
    
    def get_parents(self, node: str) -> List[str]:
        """
        Get the parents (direct causes) of a node.
        
        Args:
            node: Target node
            
        Returns:
            List of parent nodes
        """
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """
        Get the children (direct effects) of a node.
        
        Args:
            node: Source node
            
        Returns:
            List of child nodes
        """
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> Set[str]:
        """
        Get all ancestors (direct and indirect causes) of a node.
        
        Args:
            node: Target node
            
        Returns:
            Set of ancestor nodes
        """
        ancestors = set()
        
        def find_ancestors(current):
            for parent in self.get_parents(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    find_ancestors(parent)
        
        find_ancestors(node)
        return ancestors
    
    def get_descendants(self, node: str) -> Set[str]:
        """
        Get all descendants (direct and indirect effects) of a node.
        
        Args:
            node: Source node
            
        Returns:
            Set of descendant nodes
        """
        descendants = set()
        
        def find_descendants(current):
            for child in self.get_children(current):
                if child not in descendants:
                    descendants.add(child)
                    find_descendants(child)
        
        find_descendants(node)
        return descendants
    
    def is_valid_dag(self) -> bool:
        """
        Check if the graph is a valid directed acyclic graph (DAG).
        
        Returns:
            True if the graph is a DAG, False otherwise
        """
        return nx.is_directed_acyclic_graph(self.graph)
    
    def get_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Get all directed paths from source to target.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            List of paths, where each path is a list of nodes
        """
        if not nx.has_path(self.graph, source, target):
            return []
        
        paths = []
        for path in nx.all_simple_paths(self.graph, source, target):
            paths.append(path)
        
        return paths
    
    def get_mediators(self, source: str, target: str) -> Set[str]:
        """
        Get all mediator nodes between source and target.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            Set of mediator nodes
        """
        mediators = set()
        
        for path in self.get_paths(source, target):
            # Exclude source and target from mediators
            mediators.update(path[1:-1])
        
        return mediators
    
    def discover_from_data(self, 
                          X: pd.DataFrame, 
                          s_idx: int,
                          correlation_threshold: float = 0.05,
                          partial_correlation_threshold: float = 0.03,
                          outcome_idx: Optional[int] = None) -> None:
        """
        Discover causal structure from data using correlation analysis and domain knowledge.
        
        Args:
            X: Feature matrix as a pandas DataFrame
            s_idx: Index of the protected attribute (gender)
            correlation_threshold: Threshold for considering correlations significant
            partial_correlation_threshold: Threshold for partial correlations
            outcome_idx: Index of the outcome variable (if available)
        """
        # Reset graph
        self.graph = nx.DiGraph()
        
        # Get features names from DataFrame
        feature_names = X.columns.tolist()
        protected_attribute = feature_names[s_idx]
        self.protected_attribute = protected_attribute
        
        # Add nodes
        self.graph.add_nodes_from(feature_names)
        
        # Compute correlation matrix
        corr_matrix = X.corr()
        
        # First, add edges from protected attribute to correlated features
        for i, feature in enumerate(feature_names):
            if i != s_idx and abs(corr_matrix.iloc[s_idx, i]) > correlation_threshold:
                self.graph.add_edge(protected_attribute, feature)
        
        # Then, add edges between other variables based on correlation and causality assumptions
        for i, j in combinations(range(len(feature_names)), 2):
            if i == s_idx or j == s_idx:
                continue  # Already handled protected attribute
                
            feature_i = feature_names[i]
            feature_j = feature_names[j]
            
            corr = abs(corr_matrix.iloc[i, j])
            
            if corr > correlation_threshold:
                # Use partial correlation to determine direction
                
                # Partial correlation controlling for protected attribute
                partial_corr_i_s = abs(corr_matrix.iloc[i, s_idx])
                partial_corr_j_s = abs(corr_matrix.iloc[j, s_idx])
                
                # If one variable is more strongly correlated with protected attribute,
                # assume it's a mediator between protected attribute and the other variable
                if partial_corr_i_s > partial_corr_j_s + partial_correlation_threshold:
                    self.graph.add_edge(feature_i, feature_j)
                elif partial_corr_j_s > partial_corr_i_s + partial_correlation_threshold:
                    self.graph.add_edge(feature_j, feature_i)
                else:
                    # If no clear direction, use heuristic (e.g., lower index is earlier in causal order)
                    self.graph.add_edge(feature_i, feature_j)
        
        # If outcome is provided, ensure it has no outgoing edges
        if outcome_idx is not None:
            outcome = feature_names[outcome_idx]
            for edge in list(self.graph.out_edges(outcome)):
                self.graph.remove_edge(edge[0], edge[1])
        
        # Make sure the graph is acyclic by removing edges that create cycles
        while not self.is_valid_dag():
            # Find a cycle
            try:
                cycle = nx.find_cycle(self.graph, orientation="original")
                # Remove the last edge in the cycle
                self.graph.remove_edge(cycle[-1][0], cycle[-1][1])
            except nx.NetworkXNoCycle:
                break
        
        # Categorize features based on their relationship to protected attribute
        self._categorize_features()
    
    def discover_from_feature_categories(self,
                                       feature_names: List[str],
                                       protected_attribute: str,
                                       direct_features: List[str],
                                       proxy_features: List[str],
                                       mediator_features: List[str],
                                       neutral_features: List[str],
                                       outcome: Optional[str] = None) -> None:
        """
        Create a causal graph based on pre-categorized features.
        
        Args:
            feature_names: List of all feature names
            protected_attribute: Name of the protected attribute
            direct_features: Features directly affected by the protected attribute only
            proxy_features: Features that can serve as proxies for the protected attribute
            mediator_features: Features that mediate between protected attribute and outcome
            neutral_features: Features unrelated to the protected attribute
            outcome: Name of the outcome variable (if available)
        """
        # Reset graph
        self.graph = nx.DiGraph()
        self.protected_attribute = protected_attribute
        
        # Add all nodes
        self.graph.add_nodes_from(feature_names)
        
        # Add protected attribute if not already in feature_names
        if protected_attribute not in feature_names:
            self.graph.add_node(protected_attribute)
        
        # Add outcome if provided and not already in feature_names
        if outcome is not None and outcome not in feature_names:
            self.graph.add_node(outcome)
        
        # Add edges from protected attribute to direct features
        for feature in direct_features:
            self.graph.add_edge(protected_attribute, feature)
        
        # Add edges for proxy features (bidirectional relationship with protected attribute)
        for feature in proxy_features:
            self.graph.add_edge(feature, protected_attribute)
        
        # Add edges for mediator features
        for feature in mediator_features:
            self.graph.add_edge(protected_attribute, feature)
            if outcome is not None:
                self.graph.add_edge(feature, outcome)
        
        # For neutral features, no edges to/from protected attribute
        # But they might influence the outcome
        if outcome is not None:
            for feature in neutral_features:
                self.graph.add_edge(feature, outcome)
        
        # Store feature categories
        self.direct_features = set(direct_features)
        self.proxy_features = set(proxy_features)
        self.mediator_features = set(mediator_features)
        self.neutral_features = set(neutral_features)
    
    def _categorize_features(self) -> None:
        """
        Categorize features based on their relationship to the protected attribute.
        """
        # Reset categories
        self.direct_features = set()
        self.proxy_features = set()
        self.mediator_features = set()
        self.neutral_features = set()
        
        all_features = set(self.graph.nodes())
        outcome_candidates = {node for node in all_features if len(list(self.graph.successors(node))) == 0}
        
        # Filter out the most likely outcome node if multiple candidates
        outcome = None
        if len(outcome_candidates) > 1:
            # The node with the most incoming edges is likely the outcome
            outcome = max(outcome_candidates, key=lambda node: len(list(self.graph.predecessors(node))))
            outcome_candidates = {outcome}
        elif len(outcome_candidates) == 1:
            outcome = next(iter(outcome_candidates))
        
        for feature in all_features:
            if feature == self.protected_attribute or (outcome is not None and feature == outcome):
                continue
                
            # Direct features: direct children of protected attribute
            if self.protected_attribute in self.get_parents(feature):
                # If also leads to outcome, it's a mediator
                if outcome is not None and nx.has_path(self.graph, feature, outcome):
                    self.mediator_features.add(feature)
                else:
                    self.direct_features.add(feature)
            
            # Proxy features: direct parents of protected attribute
            elif self.protected_attribute in self.get_children(feature):
                self.proxy_features.add(feature)
            
            # Neutral features: no direct connection to protected attribute
            else:
                self.neutral_features.add(feature)
    
    def visualize(self, 
                 figsize: Tuple[int, int] = (12, 8), 
                 node_size: int = 1000,
                 node_colors: Dict[str, str] = None,
                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the causal graph.
        
        Args:
            figsize: Size of the figure
            node_size: Size of nodes in the graph
            node_colors: Dictionary mapping categories to colors
            save_path: If provided, save the visualization to this path
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Default node colors by category
        if node_colors is None:
            node_colors = {
                'protected': 'orangered',
                'direct': 'gold',
                'proxy': 'lightcoral',
                'mediator': 'lightseagreen',
                'neutral': 'lightskyblue',
                'outcome': 'mediumpurple',
                'other': 'lightgray'
            }
        
        # Determine node color based on its category
        node_color_map = {}
        for node in self.graph.nodes():
            if node == self.protected_attribute:
                node_color_map[node] = node_colors['protected']
            elif node in self.direct_features:
                node_color_map[node] = node_colors['direct']
            elif node in self.proxy_features:
                node_color_map[node] = node_colors['proxy']
            elif node in self.mediator_features:
                node_color_map[node] = node_colors['mediator']
            elif node in self.neutral_features:
                node_color_map[node] = node_colors['neutral']
            else:
                # Assume any node with no outgoing edges is an outcome
                if len(list(self.graph.successors(node))) == 0:
                    node_color_map[node] = node_colors['outcome']
                else:
                    node_color_map[node] = node_colors['other']
        
        # Prepare layout
        layout = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, 
            layout, 
            node_color=[node_color_map.get(node, 'gray') for node in self.graph.nodes()],
            node_size=node_size,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, 
            layout,
            arrowstyle='-|>',
            arrowsize=20,
            alpha=0.6,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, 
            layout, 
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cat)
            for cat, color in node_colors.items()
            if any(value == color for value in node_color_map.values())
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and remove axis
        plt.title('Causal Graph Visualization', fontsize=16)
        plt.axis('off')
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def get_structural_equations(self, 
                                X: pd.DataFrame,
                                models: Dict[str, str] = None) -> Dict[str, object]:
        """
        Generate structural equations for features based on their parents in the causal graph.
        
        Args:
            X: Feature data as a pandas DataFrame
            models: Dictionary mapping feature names to model types ('linear' or 'random_forest')
            
        Returns:
            Dictionary mapping feature names to fitted model objects
        """
        if models is None:
            models = {}
        
        structural_equations = {}
        
        # Iterate through nodes in topological order (parents before children)
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            
            # Skip if no parents
            if not parents:
                continue
            
            # Skip if node not in DataFrame (e.g., outcome variable)
            if node not in X.columns:
                continue
            
            # Filter parents to only those in DataFrame
            parents = [p for p in parents if p in X.columns]
            
            # Skip if no valid parents
            if not parents:
                continue
            
            # Prepare data
            X_parents = X[parents]
            y = X[node]
            
            # Select model type
            model_type = models.get(node, 'linear')
            
            # Fit model based on data type and specified model type
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()
            
            # Fit model
            model.fit(X_parents, y)
            
            # Store model
            structural_equations[node] = model
        
        return structural_equations
    
    def save(self, filepath: str) -> None:
        """
        Save the causal graph to a file.
        
        Args:
            filepath: Path to save the graph
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'protected_attribute': self.protected_attribute,
                'direct_features': self.direct_features,
                'proxy_features': self.proxy_features,
                'mediator_features': self.mediator_features,
                'neutral_features': self.neutral_features
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'CausalGraph':
        """
        Load a causal graph from a file.
        
        Args:
            filepath: Path to the saved graph
            
        Returns:
            CausalGraph object
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        causal_graph = cls()
        causal_graph.graph = data['graph']
        causal_graph.protected_attribute = data['protected_attribute']
        causal_graph.direct_features = data['direct_features']
        causal_graph.proxy_features = data['proxy_features']
        causal_graph.mediator_features = data['mediator_features']
        causal_graph.neutral_features = data['neutral_features']
        
        return causal_graph


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Gender (protected attribute)
    gender = np.random.binomial(1, 0.5, n_samples)  # 0=male, 1=female
    
    # Direct effects of gender
    age = 40 + gender * (-5) + np.random.normal(0, 10, n_samples)  # Women slightly younger on average
    education = 12 + gender * (1) + np.random.normal(0, 3, n_samples)  # Women slightly more educated
    
    # Mediator variables (affected by gender and affecting income)
    occupation = 5 + 0.2 * education - 1 * gender + np.random.normal(0, 2, n_samples)  # Gender affects occupation
    experience = age - education - 5 + np.random.normal(0, 3, n_samples)  # Function of age and education
    
    # Neutral variables (not affected by gender)
    region = np.random.randint(0, 4, n_samples)  # Region code (0-3)
    
    # Outcome: income (affected by education, occupation, experience, and region)
    income = (
        30000 + 5000 * education + 3000 * occupation + 1000 * experience + 
        2000 * region - 10000 * gender +  # Gender has direct effect on income (gender pay gap)
        np.random.normal(0, 10000, n_samples)
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'age': age,
        'education': education,
        'occupation': occupation,
        'experience': experience,
        'region': region,
        'income': income
    })
    
    # Create and visualize causal graph
    causal_graph = CausalGraph()
    causal_graph.discover_from_data(
        df, s_idx=0, correlation_threshold=0.05, outcome_idx=6
    )
    
    # Visualize
    causal_graph.visualize()
    # plt.show()
    plt.close()