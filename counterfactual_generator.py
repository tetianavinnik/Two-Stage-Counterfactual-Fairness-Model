"""
Counterfactual generation module for the Two-Stage Counterfactual Fairness Model (TSCFM).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from causal_graph import CausalGraph
import logging

# Set up logging
logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Generates counterfactual data based on causal relationships.
    """
    
    def __init__(self, 
                causal_graph: CausalGraph,
                method: str = "structural_equation",
                adjustment_strength: float = 1.0,
                random_state: int = 42,
                amplification_factor: float = 1.5):
        """
        Initialize the counterfactual generator.
        
        Args:
            causal_graph: Causal graph representing relationships between features
            method: Method to use for generating counterfactuals
                    ('structural_equation', 'matching', or 'generative')
            adjustment_strength: Strength of counterfactual adjustments (0.0-2.0)
            random_state: Random seed for reproducibility
            amplification_factor: Factor to amplify counterfactual changes
        """
        self.causal_graph = causal_graph
        self.method = method.lower()
        self.adjustment_strength = adjustment_strength
        self.random_state = random_state
        self.amplification_factor = amplification_factor
        self.structural_equations = {}
        self.feature_distributions = {}
        
        # Set random seed
        np.random.seed(self.random_state)
        
        # Validate method
        valid_methods = ["structural_equation", "matching", "generative"]
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Must be one of {valid_methods}")
    
    def fit(self, X: np.ndarray, s: np.ndarray, feature_names: List[str]) -> 'CounterfactualGenerator':
        """
        Fit the counterfactual generator to the data.
        
        Args:
            X: Feature matrix
            s: Protected attribute values
            feature_names: Names of features
            
        Returns:
            Self for method chaining
        """
        # Ensure consistent random state
        np.random.seed(self.random_state)
        
        # Create DataFrame for easier handling
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add protected attribute
        protected_attr = self.causal_graph.protected_attribute
        df[protected_attr] = s
        
        # Enhance causal structure if needed (deterministically)
        self._enhance_causal_structure(df)
        
        # Perform method-specific fitting
        if self.method == "structural_equation":
            self._fit_structural_equations(df)
        elif self.method == "matching":
            self._fit_matching(df, protected_attr)
        elif self.method == "generative":
            self._fit_generative(df, protected_attr)
        
        return self
    

    def _enhance_causal_structure(self, df=None):
        """
        Enhance the causal structure to create stronger counterfactual effects 
        by selecting the most influential neutral features to convert to mediators.
        """
        # Get protected attribute
        protected_attr = self.causal_graph.protected_attribute
        
        # If no mediator features are detected, convert some neutral features to mediators
        if len(self.causal_graph.mediator_features) == 0 and len(self.causal_graph.neutral_features) > 0:
            neutral_features = list(self.causal_graph.neutral_features)
            
            # If df is provided, use correlation to select features
            if df is not None and protected_attr in df.columns:
                # Calculate correlations between each feature and protected attribute
                correlations = []
                for feature in neutral_features:
                    if feature in df.columns:
                        corr = abs(np.corrcoef(df[feature].values, df[protected_attr].values)[0, 1])
                        correlations.append((feature, corr))
                
                # Sort by correlation (highest first)
                sorted_features = [f for f, corr in sorted(correlations, key=lambda x: x[1], reverse=True)]
            else:
                # If no dataframe available, use deterministic sorting
                sorted_features = sorted(neutral_features)
            
            # Convert up to 3 features with highest correlation
            num_to_convert = min(3, len(sorted_features))
            features_to_convert = sorted_features[:num_to_convert]
            
            logger.info(f"Enhancing causal structure: Converting {num_to_convert} neutral features to mediators")
            
            # Update feature categories in causal graph
            for feature in features_to_convert:
                self.causal_graph.neutral_features.remove(feature)
                self.causal_graph.mediator_features.add(feature)
                
                # Add edge from protected attribute to this feature in the graph
                if not self.causal_graph.graph.has_edge(protected_attr, feature):
                    self.causal_graph.graph.add_edge(protected_attr, feature)
                    logger.info(f"Added causal edge: {protected_attr} -> {feature}")
    
    def _fit_structural_equations(self, df: pd.DataFrame) -> None:
        """
        Fit structural equations for each feature based on its parents in the causal graph.
        
        Args:
            df: DataFrame with features and protected attribute
        """
        logger.info("Fitting structural equations...")
        
        # Get feature names (excluding protected attribute)
        features = [col for col in df.columns if col != self.causal_graph.protected_attribute]
        
        # Get protected attribute
        protected_attr = self.causal_graph.protected_attribute
        
        # Iterate through features
        for feature in features:
            # Get parents of this feature in causal graph
            parents = self.causal_graph.get_parents(feature)
            
            # Skip if feature has no parents
            if not parents:
                logger.info(f"Feature {feature} has no parents, skipping")
                continue
            
            # Skip if protected attribute is not a parent or ancestor (no causal effect from protected attribute)
            if protected_attr not in parents and protected_attr not in self.causal_graph.get_ancestors(feature):
                logger.info(f"Feature {feature} not affected by {protected_attr}, skipping")
                continue
            
            # Get parent features
            X_parents = df[parents]
            y_feature = df[feature]
                   
            
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            
            # Fit model
            model.fit(X_parents, y_feature)
            
            # Store model and column information
            self.structural_equations[feature] = {
                'model': model,
                'parents': parents
            }
            
            logger.info(f"Fitted structural equation for {feature} using {parents}")
        
        # Also store feature distributions for each protected attribute value
        for s_val in [0, 1]:
            self.feature_distributions[s_val] = {
                feature: {
                    'mean': df.loc[df[protected_attr] == s_val, feature].mean(),
                    'std': max(df.loc[df[protected_attr] == s_val, feature].std(), 1e-6),  # Avoid zero std
                    'min': df.loc[df[protected_attr] == s_val, feature].min(),
                    'max': df.loc[df[protected_attr] == s_val, feature].max()
                }
                for feature in features
            }
    
    def _fit_matching(self, df: pd.DataFrame, protected_attr: str) -> None:
        """
        Fit nearest neighbors model for counterfactual matching.
        
        Args:
            df: DataFrame with features and protected attribute
            protected_attr: Name of protected attribute
        """
        logger.info("Fitting matching model...")
        
        # Create separate models for each protected attribute value
        self.matching_models = {}
        self.group_indices = {}  # Store indices for each group
        
        # Get features (excluding protected attribute)
        features = [col for col in df.columns if col != protected_attr]
        
        # Initialize standardizer
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df[features])
        
        # Store dataframe for reference during transform
        self.original_df = df.copy()
        
        # Split data by protected attribute
        for s_val in [0, 1]:
            indices = df[protected_attr] == s_val
            self.group_indices[s_val] = df.index[indices].tolist()
            
            # Skip if one group has insufficient samples
            if np.sum(indices) < 2:
                logger.warning(f"Group {s_val} has insufficient samples for matching. Matching may not work properly.")
                continue
                
            X_group = X_scaled[indices]
            
            # Fit nearest neighbors model
            nn_model = NearestNeighbors(
                n_neighbors=min(5, np.sum(indices)), 
                algorithm='ball_tree'
            )
            nn_model.fit(X_group)
            
            self.matching_models[s_val] = nn_model
            
            # Store feature distributions
            self.feature_distributions[s_val] = {
                feature: {
                    'mean': df.loc[indices, feature].mean(),
                    'std': max(df.loc[indices, feature].std(), 1e-6),  # Avoid zero std
                    'min': df.loc[indices, feature].min(),
                    'max': df.loc[indices, feature].max()
                }
                for feature in features
            }
        
        logger.info("Matching models fitted")
    
    def _fit_generative(self, df: pd.DataFrame, protected_attr: str) -> None:
        """
        Fit generative model for counterfactual generation.
        
        Args:
            df: DataFrame with features and protected attribute
            protected_attr: Name of protected attribute
        """
        logger.info("Fitting generative model...")
        
        # Get feature names (excluding protected attribute)
        features = [col for col in df.columns if col != protected_attr]
        
        # Split data by protected attribute
        for s_val in [0, 1]:
            indices = df[protected_attr] == s_val
            
            # Store feature distributions
            self.feature_distributions[s_val] = {
                feature: {
                    'mean': df.loc[indices, feature].mean(),
                    'std': max(df.loc[indices, feature].std(), 1e-6),  # Avoid zero std
                    'min': df.loc[indices, feature].min(),
                    'max': df.loc[indices, feature].max()
                }
                for feature in features
            }
        
        # Analyze differences between groups
        self.feature_diffs = {}
        for feature in features:
            # Check if feature should be affected based on causal structure - no randomness
            is_affected = (feature in self.causal_graph.direct_features or 
                           feature in self.causal_graph.mediator_features or
                           feature in self.causal_graph.proxy_features)
            
            
            if is_affected:
                # Calculate difference between groups
                mean_0 = self.feature_distributions[0][feature]['mean']
                mean_1 = self.feature_distributions[1][feature]['mean']
                
                # Apply amplification consistently
                diff = (mean_1 - mean_0) * self.amplification_factor
                
                # Store difference
                self.feature_diffs[feature] = diff
        
        logger.info(f"Generative model fitted with {len(self.feature_diffs)} affected features")
    
    def validate_counterfactuals(self, X, X_cf, s, feature_names):
        """
        Validate that counterfactuals are reasonable and different from originals.
        
        Args:
            X: Original feature matrix
            X_cf: Counterfactual feature matrix
            s: Protected attribute values
            feature_names: Feature names for logging
            
        Returns:
            True if counterfactuals pass validation, False otherwise
        """
        # Calculate average absolute change
        feature_changes = np.mean(np.abs(X - X_cf), axis=0)
        mean_abs_change = np.mean(feature_changes)
        
        # Check if changes are too small
        if mean_abs_change < 0.001:
            logger.warning("Counterfactuals are almost identical to original features (avg change < 0.001)")
            return False
        
        # Check if changes are unreasonably large
        if mean_abs_change > 0.5:
            logger.warning("Counterfactuals show very large changes (avg change > 0.5)")
            
            # Log the top changed features
            top_indices = np.argsort(-feature_changes)[:5]
            
            logger.warning("Top changed features:")
            for idx in top_indices:
                logger.warning(f"  {feature_names[idx]}: {feature_changes[idx]:.4f}")
        
        # Check distribution of changes across groups
        group_0_change = np.mean(np.abs(X[s == 0] - X_cf[s == 0]))
        group_1_change = np.mean(np.abs(X[s == 1] - X_cf[s == 1]))
        group_ratio = max(group_0_change, group_1_change) / (min(group_0_change, group_1_change) + 1e-6)
        
        if group_ratio > 5:
            logger.warning(f"Imbalanced changes between groups: G0={group_0_change:.4f}, G1={group_1_change:.4f}")
        
        return True

    def transform(self, X, s, feature_names):
        """
        Generate counterfactual versions of the data by flipping the protected attribute.
        
        Args:
            X: Feature matrix
            s: Protected attribute values
            feature_names: Names of features
            
        Returns:
            Counterfactual feature matrix
        """
        # Ensure consistent random state
        np.random.seed(self.random_state)
        
        # Create DataFrame for easier handling
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add protected attribute
        protected_attr = self.causal_graph.protected_attribute
        df[protected_attr] = s
        
        # Generate counterfactuals using the specified method
        if self.method == "structural_equation":
            df_cf = self._transform_structural_equations(df, protected_attr)
        elif self.method == "matching":
            df_cf = self._transform_matching(df, protected_attr)
        elif self.method == "generative":
            df_cf = self._transform_generative(df, protected_attr)
        
        # Return counterfactual features (excluding protected attribute)
        features = [col for col in df.columns if col != protected_attr]
        X_cf = df_cf[features].values
        
        # Validate counterfactuals
        self.validate_counterfactuals(X, X_cf, s, feature_names)
        
        return X_cf
    
    def _transform_structural_equations(self, df: pd.DataFrame, protected_attr: str) -> pd.DataFrame:
        """
        Generate counterfactuals using structural equations.
        
        Args:
            df: DataFrame with features and protected attribute
            protected_attr: Name of protected attribute
            
        Returns:
            DataFrame with counterfactual features
        """
        logger.info("Generating counterfactuals using structural equations...")
        
        # Create copy of DataFrame for counterfactuals
        df_cf = df.copy()
        
        # Flip protected attribute
        df_cf[protected_attr] = 1 - df_cf[protected_attr]
        
        # Get features in topological order (parents before children)
        import networkx as nx
        try:
            features_topo = [n for n in nx.topological_sort(self.causal_graph.graph) 
                           if n != protected_attr]
        except nx.NetworkXUnfeasible:
            # If graph is not a DAG, just use features as is
            features_topo = [col for col in df.columns if col != protected_attr]
        
        # Update each feature based on structural equations
        for feature in features_topo:
            # Skip if feature has no structural equation
            if feature not in self.structural_equations:
                continue
            
            # Get structural equation info
            eq_info = self.structural_equations[feature]
            model = eq_info['model']
            parents = eq_info['parents']
            
            # Get parent features for prediction
            X_parents = df_cf[parents]

            new_values = model.predict(X_parents)
            
            # Calculate difference between predicted and original values
            original = df[feature].values
            difference = new_values - original
            
            # Apply amplified adjustment - consistently using amplification_factor
            new_values = original + difference * self.amplification_factor
            
            # Apply clamping between min and max if necessary
            if feature in self.feature_distributions[0] and feature in self.feature_distributions[1]:
                min_val = min(self.feature_distributions[0][feature]['min'], 
                                self.feature_distributions[1][feature]['min'])
                max_val = max(self.feature_distributions[0][feature]['max'], 
                                self.feature_distributions[1][feature]['max'])
                new_values = np.clip(new_values, min_val, max_val)
            
            # Update feature values
            df_cf[feature] = new_values
        
        return df_cf
    
    def _transform_matching(self, df: pd.DataFrame, protected_attr: str) -> pd.DataFrame:
        """
        Generate counterfactuals using matching.
        
        Args:
            df: DataFrame with features and protected attribute
            protected_attr: Name of protected attribute
            
        Returns:
            DataFrame with counterfactual features
        """
        logger.info("Generating counterfactuals using matching...")
        
        # Create copy of DataFrame for counterfactuals
        df_cf = df.copy()
        
        # Get features (excluding protected attribute)
        features = [col for col in df.columns if col != protected_attr]
        
        # Scale features
        X_scaled = self.scaler.transform(df[features])
        
        # Generate counterfactuals for each instance
        for i in range(len(df)):
            # Get protected attribute value
            s_val = df.iloc[i][protected_attr]
            
            # Get opposite group value
            opposite_group = 1 - s_val
            
            # Skip if matching model doesn't exist for opposite group (e.g., no samples)
            if opposite_group not in self.matching_models:
                logger.warning(f"No matching model for group {opposite_group}. Keeping original values.")
                continue
            
            # Get the nearest neighbors model for opposite group
            nn_model = self.matching_models[opposite_group]
            
            try:
                # Find nearest neighbors in opposite group
                distances, indices = nn_model.kneighbors(X_scaled[i].reshape(1, -1))
                
                # Get indices of original dataframe for opposite group
                opposite_group_indices = self.group_indices[opposite_group]
                
                # Ensure we don't go out of bounds
                if len(indices[0]) > 0 and indices[0][0] < len(opposite_group_indices):
                    # Map back to original index in the dataset
                    nearest_idx = opposite_group_indices[indices[0][0]]
                    
                    # Apply counterfactual values with consistent adjustment strength
                    for feature in features:
                        # Skip features not affected by protected attribute
                        if feature not in self.causal_graph.direct_features and \
                           feature not in self.causal_graph.mediator_features and \
                           feature not in self.causal_graph.proxy_features:
                            continue
                            
                        # Get counterfactual value from nearest neighbor
                        if nearest_idx < len(self.original_df):
                            cf_value = self.original_df.loc[nearest_idx, feature]
                            original_value = df.iloc[i][feature]
                            
                            # Calculate adjustment with consistent amplification
                            difference = cf_value - original_value
                            adjusted_value = original_value + difference * self.amplification_factor
                            
                            # Update counterfactual
                            df_cf.iloc[i, df_cf.columns.get_loc(feature)] = adjusted_value
                        else:
                            logger.warning(f"Nearest neighbor index {nearest_idx} out of bounds, using original values.")
                else:
                    logger.warning(f"No valid nearest neighbors found for sample {i}, using original values.")
            except Exception as e:
                logger.warning(f"Error finding nearest neighbors for sample {i}: {e}. Using original values.")
        
        # Flip protected attribute
        df_cf[protected_attr] = 1 - df_cf[protected_attr]
        
        return df_cf
    
    def _transform_generative(self, df: pd.DataFrame, protected_attr: str) -> pd.DataFrame:
        """
        Generate counterfactuals using a generative approach based on feature differences.
        
        Args:
            df: DataFrame with features and protected attribute
            protected_attr: Name of protected attribute
            
        Returns:
            DataFrame with counterfactual features
        """
        logger.info("Generating counterfactuals using generative approach...")
        
        # Create copy of DataFrame for counterfactuals
        df_cf = df.copy()
        
        # Get features (excluding protected attribute)
        features = [col for col in df.columns if col != protected_attr]
        
        # Apply transformations based on feature differences
        for feature, diff in self.feature_diffs.items():
            # Apply transformation based on protected attribute
            for i in range(len(df)):
                s_val = df.iloc[i][protected_attr]
                
                # Adjust in the right direction (add for 0->1, subtract for 1->0)
                adjustment = -diff if s_val == 1 else diff
                
                # Get original value
                original_value = df.iloc[i][feature]
                
                # Apply adjustment with consistent amplification factor
                adjusted_value = original_value + adjustment
                
                # Apply constraints to keep values in reasonable range
                if feature in self.feature_distributions[0] and feature in self.feature_distributions[1]:
                    # Get min/max for target group
                    target_group = 1 - s_val
                    min_val = self.feature_distributions[target_group][feature]['min']
                    max_val = self.feature_distributions[target_group][feature]['max']
                    
                    # Constrain to range
                    adjusted_value = max(min_val, min(max_val, adjusted_value))
                
                # Update counterfactual
                df_cf.iloc[i, df_cf.columns.get_loc(feature)] = adjusted_value
        
        # Flip protected attribute
        df_cf[protected_attr] = 1 - df_cf[protected_attr]
        
        return df_cf
    
    def generate_and_evaluate(self, 
                              X: np.ndarray, 
                              s: np.ndarray, 
                              feature_names: List[str],
                              output_dir: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Generate counterfactuals and evaluate them.
        
        Args:
            X: Feature matrix
            s: Protected attribute values
            feature_names: Names of features
            output_dir: Directory to save visualizations
            
        Returns:
            Tuple of (counterfactual features, evaluation metrics)
        """
        # Generate counterfactuals
        X_cf = self.transform(X, s, feature_names)
        
        # Calculate evaluation metrics
        metrics = self._evaluate_counterfactuals(X, X_cf, s, feature_names)
        
        # Generate visualizations if output directory provided
        # if output_dir:
        #     self._visualize_counterfactuals(X, X_cf, s, feature_names, output_dir)
        
        return X_cf, metrics
    
    def _evaluate_counterfactuals(self, 
                                X: np.ndarray, 
                                X_cf: np.ndarray, 
                                s: np.ndarray,
                                feature_names: List[str]) -> Dict:
        """
        Evaluate the quality of generated counterfactuals.
        
        Args:
            X: Original feature matrix
            X_cf: Counterfactual feature matrix
            s: Protected attribute values
            feature_names: Names of features
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create DataFrames for easier handling
        df = pd.DataFrame(X, columns=feature_names)
        df_cf = pd.DataFrame(X_cf, columns=feature_names)
        
        # Calculate average change per feature
        feature_changes = {}
        for feature in feature_names:
            absolute_change = np.abs(df[feature] - df_cf[feature]).mean()
            relative_change = absolute_change / (df[feature].std() + 1e-8)  # Avoid div by zero
            
            feature_changes[feature] = {
                'absolute_change': float(absolute_change),  # Convert to native Python type
                'relative_change': float(relative_change)
            }
        
        # Calculate overall change
        overall_change = np.mean([f['relative_change'] for f in feature_changes.values()])
        
        # Calculate changes by group
        group_0_change = np.mean([np.abs(df.loc[s == 0, feature] - df_cf.loc[s == 0, feature]).mean() 
                                 for feature in feature_names])
        group_1_change = np.mean([np.abs(df.loc[s == 1, feature] - df_cf.loc[s == 1, feature]).mean() 
                                 for feature in feature_names])
        
        # Calculate changes for different feature categories
        direct_features = [f for f in feature_names if f in self.causal_graph.direct_features]
        mediator_features = [f for f in feature_names if f in self.causal_graph.mediator_features]
        proxy_features = [f for f in feature_names if f in self.causal_graph.proxy_features]
        neutral_features = [f for f in feature_names if f in self.causal_graph.neutral_features]
        
        direct_change = np.mean([feature_changes[f]['relative_change'] for f in direct_features]) if direct_features else 0
        mediator_change = np.mean([feature_changes[f]['relative_change'] for f in mediator_features]) if mediator_features else 0
        proxy_change = np.mean([feature_changes[f]['relative_change'] for f in proxy_features]) if proxy_features else 0
        neutral_change = np.mean([feature_changes[f]['relative_change'] for f in neutral_features]) if neutral_features else 0
        
        # Combine metrics
        metrics = {
            'overall_change': float(overall_change),
            'group_0_change': float(group_0_change),
            'group_1_change': float(group_1_change),
            'direct_change': float(direct_change),
            'mediator_change': float(mediator_change),
            'proxy_change': float(proxy_change),
            'neutral_change': float(neutral_change),
            'feature_changes': feature_changes
        }
        
        return metrics
    
    def _visualize_counterfactuals(self, 
                             X: np.ndarray, 
                             X_cf: np.ndarray, 
                             s: np.ndarray,
                             feature_names: List[str],
                             output_dir: str) -> None:
        """
        Generate visualizations of counterfactuals.
        
        Args:
            X: Original feature matrix
            X_cf: Counterfactual feature matrix
            s: Protected attribute values
            feature_names: Names of features
            output_dir: Directory to save visualizations
        """
        # Import visualization functions
        import os
        import visualization as viz
        
        try:
            
            # Plot feature changes
            metrics = self._evaluate_counterfactuals(X, X_cf, s, feature_names)
            
            # Get top changed features
            changes = [(feature, metrics['feature_changes'][feature]['relative_change']) 
                    for feature in feature_names]
            changes.sort(key=lambda x: x[1], reverse=True)
            
            # Plot top 20 changes
            fig, ax = plt.subplots(figsize=(12, 8))
            features_to_plot = changes[:min(20, len(changes))]
            if features_to_plot:
                plot_features, plot_values = zip(*features_to_plot)
                ax.bar(range(len(plot_features)), plot_values)
                ax.set_xticks(range(len(plot_features)))
                ax.set_xticklabels(plot_features, rotation=90)
                ax.set_title("Top Feature Changes in Counterfactuals")
                ax.set_ylabel("Relative Change")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{self.method}_feature_changes.png"), dpi=300)
                plt.close(fig)
            
            # Plot embedding space
            embedding_path = os.path.join(output_dir, f"{self.method}_embedding_space.png")
            embedding_fig = viz.plot_embedding_space(X, X_cf, s, method='pca', save_path=embedding_path)
            if embedding_fig is not None:
                plt.close(embedding_fig)
                
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
    
    def save(self, filepath: str) -> None:
        """
        Save the counterfactual generator to a file.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'adjustment_strength': self.adjustment_strength,
                'random_state': self.random_state,
                'amplification_factor': self.amplification_factor,
                'structural_equations': self.structural_equations,
                'feature_distributions': self.feature_distributions,
                'feature_diffs': getattr(self, 'feature_diffs', {}),
                'causal_graph': self.causal_graph
            }, f)
    
    @classmethod
    def load(cls, filepath: str, causal_graph: Optional[CausalGraph] = None) -> 'CounterfactualGenerator':
        """
        Load a counterfactual generator from a file.
        
        Args:
            filepath: Path to the saved model
            causal_graph: Causal graph (if not included in saved model)
            
        Returns:
            Loaded CounterfactualGenerator
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Use provided causal graph or the one from the saved model
        if causal_graph is None:
            causal_graph = data['causal_graph']
        
        # Create instance
        generator = cls(
            causal_graph=causal_graph,
            method=data['method'],
            adjustment_strength=data['adjustment_strength'],
            random_state=data['random_state'],
            amplification_factor=data.get('amplification_factor', 1.0)  # Default if not in saved data
        )
        
        # Restore saved state
        generator.structural_equations = data['structural_equations']
        generator.feature_distributions = data['feature_distributions']
        if 'feature_diffs' in data:
            generator.feature_diffs = data['feature_diffs']
        
        return generator