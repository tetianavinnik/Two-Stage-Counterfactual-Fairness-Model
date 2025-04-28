"""
Two-Stage Counterfactual Fairness Model (TSCFM) implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import os
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import time
from scipy.optimize import minimize

# Set up logging
logger = logging.getLogger(__name__)


class TSCFM(BaseEstimator, ClassifierMixin):
    """
    Two-Stage Counterfactual Fairness Model for fair credit scoring.
    
    Key improvements over standard TSCFM:
    1. No dependency on true labels during prediction time
    2. Calibrated fairness adjustments that respect parameter settings
    3. Better fairness-accuracy trade-off through stratified adjustments
    4.  verification of fairness improvements
    5. Specialized optimization for different fairness metrics
    """
    
    def __init__(self, 
                base_model_type: str = "random_forest",
                base_model_params: Optional[Dict] = None,
                counterfactual_method: str = "structural_equation",
                adjustment_strength: float = 0.8,
                fairness_constraint: str = "demographic_parity_difference",
                fairness_threshold: float = 0.05,
                amplification_factor: float = 1.5,
                random_state: int = 42,
                fairness_weight: float = 0.7,
                prediction_threshold: float = 0.5,
                verbose: bool = False):
        """
        Initialize the  TSCFM model.
        
        Args:
            base_model_type: Type of base classifier ('random_forest', 'logistic_regression', etc.)
            base_model_params: Parameters for the base classifier
            counterfactual_method: Method for generating counterfactuals 
                                 ('structural_equation', 'matching', 'generative')
            adjustment_strength: Strength of counterfactual adjustments
                               Lower values favor accuracy, higher values favor fairness
            fairness_constraint: Type of fairness constraint to enforce
                                ('demographic_parity_difference', 'equalized_odds_difference', 
                                 'equal_opportunity_difference')
            fairness_threshold: Maximum allowed value for the fairness metric
            amplification_factor: Factor to amplify counterfactual changes
            random_state: Random seed for reproducibility
            fairness_weight: Weight for the fairness term in optimization (0.0-1.0)
            prediction_threshold: Threshold for converting probabilities to binary predictions
            verbose: Whether to print detailed debug information
        """
        self.base_model_type = base_model_type
        self.base_model_params = base_model_params or {}
        self.counterfactual_method = counterfactual_method
        self.adjustment_strength = adjustment_strength
        self.fairness_constraint = fairness_constraint
        self.fairness_threshold = fairness_threshold
        self.amplification_factor = amplification_factor
        self.random_state = random_state
        self.fairness_weight = fairness_weight
        self.prediction_threshold = prediction_threshold
        self.verbose = verbose
        
        # Components to be initialized during fitting
        self.base_model = None
        self.causal_graph = None
        self.counterfactual_generator = None
        self.feature_names = None
        
        # Fairness reference statistics (captured during training)
        self.fairness_statistics = {}
        self.baseline_metrics = {}
        self.group_statistics = {}
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message with the specified level.
        
        Args:
            message: The message to log
            level: The logging level ('debug', 'info', 'warning', 'error')
        """
        if level == "debug" and self.verbose:
            logger.debug(message)
            if self.verbose:
                print(f"[DEBUG] {message}")
        elif level == "info":
            logger.info(message)
            if self.verbose:
                print(f"[INFO] {message}")
        elif level == "warning":
            logger.warning(message)
            if self.verbose:
                print(f"[WARNING] {message}")
        elif level == "error":
            logger.error(message)
            if self.verbose:
                print(f"[ERROR] {message}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, s: np.ndarray, 
           feature_names: Optional[List[str]] = None,
           causal_graph: Optional[Any] = None) -> 'TSCFM':
        """
        Fit the TSCFM model to the data.
        
        Args:
            X: Feature matrix
            y: Target variable (binary)
            s: Protected attribute values (binary)
            feature_names: Names of features (if None, use default names)
            causal_graph: Pre-built causal graph (if None, discover from data)
            
        Returns:
            Self for method chaining
        """
        self._log("Fitting TSCFM model...")
        
        # Set feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names = feature_names
        
        # Step 1: Train base model
        self._log("Training base model...")
        self.base_model = self._create_base_model()
        self.base_model.fit(X, y)
        
        # Step 2: Set up causal graph
        if causal_graph is not None:
            self.causal_graph = causal_graph
            self._log("Using provided causal graph")
        else:
            self._log("Discovering causal graph from data...")
            self._discover_causal_graph(X, y, s, feature_names)
        
        # Step 3: Set up counterfactual generator
        self._log(f"Setting up counterfactual generator with method: {self.counterfactual_method}")
        self._setup_counterfactual_generator()
        self.counterfactual_generator.fit(X, s, feature_names)
        
        # Step 4: Generate baseline predictions and metrics
        baseline_preds = self.base_model.predict(X)
        baseline_probs = self.base_model.predict_proba(X)[:, 1]
        
        # Performance metrics
        self.baseline_metrics = {
            'accuracy': accuracy_score(y, baseline_preds),
            'f1_score': f1_score(y, baseline_preds),
            'roc_auc': roc_auc_score(y, baseline_probs)
        }
        
        # Step 5: Compute fairness reference statistics for later use
        self._compute_fairness_statistics(X, y, s, baseline_probs)
        
        # Log results
        self._log(f"Base model accuracy: {self.baseline_metrics['accuracy']:.4f}")
        for metric_name, metric_value in self.fairness_statistics.items():
            self._log(f"Base model {metric_name}: {metric_value:.4f}")
        
        return self
    
    def _create_base_model(self):
        """
        Create the base predictive model.
        """
        from base_models import BaseModelWrapper
        
        return BaseModelWrapper(
            model_type=self.base_model_type,
            hyperparams=self.base_model_params,
            random_state=self.random_state
        )
    
    def _discover_causal_graph(self, X, y, s, feature_names):
        """
        Discover the causal graph from data.
        """
        from causal_graph import CausalGraph
        
        # Create a DataFrame with features, target, and protected attribute
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['protected'] = s
        
        # Discover causal structure
        self.causal_graph = CausalGraph(protected_attribute='protected')
        self.causal_graph.discover_from_data(
            df, 
            s_idx=df.columns.get_loc('protected'), 
            outcome_idx=df.columns.get_loc('target'),
            correlation_threshold=0.05
        )
        
        # Enhance causal graph with more direct effects if needed
        self._enhance_causal_graph(df)
    
    def _enhance_causal_graph(self, df):
        """
        Enhance the causal graph by adding important direct effects.
        """
        protected_attr = self.causal_graph.protected_attribute
        
        # Calculate correlations with protected attribute
        protected_corrs = df.corr()[protected_attr].abs()
        
        # Get top correlated features
        sorted_features = protected_corrs.sort_values(ascending=False)
        top_features = [f for f in sorted_features.index 
                      if f not in [protected_attr, 'target']][:5]
        
        # Add edges from protected attribute to top features
        for feature in top_features:
            if not self.causal_graph.graph.has_edge(protected_attr, feature):
                self.causal_graph.graph.add_edge(protected_attr, feature)
                self.causal_graph.direct_features.add(feature)
                
                # Remove from other categories if present
                if feature in self.causal_graph.neutral_features:
                    self.causal_graph.neutral_features.remove(feature)
                if feature in self.causal_graph.proxy_features:
                    self.causal_graph.proxy_features.remove(feature)
    
    def _setup_counterfactual_generator(self):
        """
        Set up the counterfactual generator.
        """
        from counterfactual_generator import CounterfactualGenerator
        
        self.counterfactual_generator = CounterfactualGenerator(
            causal_graph=self.causal_graph,
            method=self.counterfactual_method,
            adjustment_strength=self.adjustment_strength,
            amplification_factor=self.amplification_factor,
            random_state=self.random_state
        )
    
    def _compute_fairness_statistics(self, X: np.ndarray, y: np.ndarray, 
                                  s: np.ndarray, baseline_probs: np.ndarray):
        """
        Compute fairness statistics during training for later use in prediction.
        This is the key function to avoid using true labels during prediction.
        
        Args:
            X: Feature matrix
            y: Target variable
            s: Protected attribute values
            baseline_probs: Base model prediction probabilities
        """
        self._log("Computing fairness reference statistics...")
        
        # Calculate demographic parity statistics
        mean_0 = baseline_probs[s == 0].mean()
        mean_1 = baseline_probs[s == 1].mean()
        dp_diff = abs(mean_0 - mean_1)
        
        self.fairness_statistics['demographic_parity_difference'] = dp_diff
        
        # Create positive and negative masks
        positive_mask = (y == 1)
        negative_mask = (y == 0)
        
        # Calculate equal opportunity statistics (TPR differences)
        if np.any((s == 0) & positive_mask) and np.any((s == 1) & positive_mask):
            tpr_0 = baseline_probs[(s == 0) & positive_mask].mean()
            tpr_1 = baseline_probs[(s == 1) & positive_mask].mean()
            eo_diff = abs(tpr_0 - tpr_1)
            
            # Store group-specific TPR
            self.group_statistics['tpr_0'] = tpr_0
            self.group_statistics['tpr_1'] = tpr_1
            
            self.fairness_statistics['equal_opportunity_difference'] = eo_diff
        
        # Calculate equalized odds statistics (TPR and FPR differences)
        if (np.any((s == 0) & positive_mask) and np.any((s == 1) & positive_mask) and
            np.any((s == 0) & negative_mask) and np.any((s == 1) & negative_mask)):
            
            # TPR calculations (already done above)
            if 'tpr_0' not in self.group_statistics:
                tpr_0 = baseline_probs[(s == 0) & positive_mask].mean()
                tpr_1 = baseline_probs[(s == 1) & positive_mask].mean()
                self.group_statistics['tpr_0'] = tpr_0
                self.group_statistics['tpr_1'] = tpr_1
            else:
                tpr_0 = self.group_statistics['tpr_0']
                tpr_1 = self.group_statistics['tpr_1']
            
            # FPR calculations
            fpr_0 = baseline_probs[(s == 0) & negative_mask].mean()
            fpr_1 = baseline_probs[(s == 1) & negative_mask].mean()
            
            # Store group-specific FPR
            self.group_statistics['fpr_0'] = fpr_0
            self.group_statistics['fpr_1'] = fpr_1
            
            # Calculate EOd as average of TPR and FPR differences
            tpr_diff = abs(tpr_0 - tpr_1)
            fpr_diff = abs(fpr_0 - fpr_1)
            eod_diff = (tpr_diff + fpr_diff) / 2
            
            self.fairness_statistics['equalized_odds_difference'] = eod_diff
        
        # Store overall statistics about prediction distributions by group
        for group in [0, 1]:
            group_mask = (s == group)
            
            # Overall statistics
            self.group_statistics[f'mean_{group}'] = baseline_probs[group_mask].mean()
            self.group_statistics[f'std_{group}'] = baseline_probs[group_mask].std()
            
            # Positive class statistics
            if np.any(group_mask & positive_mask):
                self.group_statistics[f'pos_mean_{group}'] = baseline_probs[group_mask & positive_mask].mean()
                self.group_statistics[f'pos_std_{group}'] = baseline_probs[group_mask & positive_mask].std()
            
            # Negative class statistics
            if np.any(group_mask & negative_mask):
                self.group_statistics[f'neg_mean_{group}'] = baseline_probs[group_mask & negative_mask].mean()
                self.group_statistics[f'neg_std_{group}'] = baseline_probs[group_mask & negative_mask].std()
    
    def predict(self, X: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Generate fair predictions using counterfactual adjustments.
        
        Args:
            X: Feature matrix
            s: Protected attribute values
            
        Returns:
            Fair binary predictions
        """
        if self.base_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get prediction probabilities
        fair_probs = self.predict_proba(X, s)[:, 1]
        
        # Convert to binary predictions using threshold
        return (fair_probs >= self.prediction_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Generate fair probability predictions using counterfactual adjustments.
        
        Args:
            X: Feature matrix
            s: Protected attribute values
            
        Returns:
            Fair probability predictions (2D array of shape [n_samples, 2])
        """
        if self.base_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Step 1: Get baseline predictions
        baseline_probs = self.base_model.predict_proba(X)[:, 1]
        
        # Step 2: Generate counterfactual features
        X_cf = self.counterfactual_generator.transform(X, s, self.feature_names)
        
        # Step 3: Get counterfactual predictions
        cf_probs = self.base_model.predict_proba(X_cf)[:, 1]
        
        # Step 4: Apply fairness adjustment based on constraint
        if self.fairness_constraint == "demographic_parity_difference":
            fair_probs = self._adjust_for_demographic_parity(baseline_probs, cf_probs, s)
        elif self.fairness_constraint == "equal_opportunity_difference":
            fair_probs = self._adjust_for_equal_opportunity(baseline_probs, cf_probs, s)
        elif self.fairness_constraint == "equalized_odds_difference":
            fair_probs = self._adjust_for_equalized_odds(baseline_probs, cf_probs, s)
        else:
            self._log(f"Unknown fairness constraint: {self.fairness_constraint}", level="warning")
            fair_probs = baseline_probs  # Default to baseline if constraint not recognized
        
        # Return probabilities as 2D array (sklearn standard)
        result = np.zeros((len(fair_probs), 2))
        result[:, 1] = fair_probs
        result[:, 0] = 1 - fair_probs
        
        return result
    
    def _adjust_for_demographic_parity(self, baseline_probs: np.ndarray, 
                                     cf_probs: np.ndarray, 
                                     s: np.ndarray) -> np.ndarray:
        """
        Adjust predictions to satisfy demographic parity.
        
        Args:
            baseline_probs: Original prediction probabilities
            cf_probs: Counterfactual prediction probabilities
            s: Protected attribute values
            
        Returns:
            Adjusted fair probabilities
        """
        self._log("Adjusting for demographic parity...")
        
        # Calculate group means for baseline predictions
        mean_0 = baseline_probs[s == 0].mean()
        mean_1 = baseline_probs[s == 1].mean()
        
        # Determine target mean (average of both groups)
        target_mean = (mean_0 + mean_1) / 2
        
        # Calculate group-specific adjustment factors
        alpha_0 = 0.5  # Default mixing weight
        alpha_1 = 0.5  # Default mixing weight
        
        # Adjust alphas based on which group needs more correction
        if mean_0 > mean_1:  # Group 0 needs to be lowered
            # How far each group is from target
            distance_0 = mean_0 - target_mean  # Positive value
            distance_1 = target_mean - mean_1  # Positive value
            
            # Calculate optimal alpha to reach the target (weighted by adjustment strength)
            cf_mean_0 = cf_probs[s == 0].mean()
            if mean_0 != cf_mean_0:  # Avoid division by zero
                alpha_0 = max(0, min(1, 1 - self.adjustment_strength * distance_0 / (mean_0 - cf_mean_0)))
            
            cf_mean_1 = cf_probs[s == 1].mean()
            if mean_1 != cf_mean_1:  # Avoid division by zero
                alpha_1 = max(0, min(1, 1 - self.adjustment_strength * distance_1 / (cf_mean_1 - mean_1)))
        else:  # Group 1 needs to be lowered
            # How far each group is from target
            distance_0 = target_mean - mean_0  # Positive value
            distance_1 = mean_1 - target_mean  # Positive value
            
            # Calculate optimal alpha to reach the target (weighted by adjustment strength)
            cf_mean_0 = cf_probs[s == 0].mean()
            if mean_0 != cf_mean_0:  # Avoid division by zero
                alpha_0 = max(0, min(1, 1 - self.adjustment_strength * distance_0 / (cf_mean_0 - mean_0)))
            
            cf_mean_1 = cf_probs[s == 1].mean()
            if mean_1 != cf_mean_1:  # Avoid division by zero
                alpha_1 = max(0, min(1, 1 - self.adjustment_strength * distance_1 / (mean_1 - cf_mean_1)))
        
        # Create result array
        result = baseline_probs.copy()
        
        # Apply group-specific adjustments
        result[s == 0] = alpha_0 * baseline_probs[s == 0] + (1 - alpha_0) * cf_probs[s == 0]
        result[s == 1] = alpha_1 * baseline_probs[s == 1] + (1 - alpha_1) * cf_probs[s == 1]
        
        # Ensure probabilities are in [0, 1]
        result = np.clip(result, 0, 1)
        
        # Log final statistics
        self._log(f"DP adjustment - alphas: [{alpha_0:.4f}, {alpha_1:.4f}]")
        self._log(f"Original means: [{mean_0:.4f}, {mean_1:.4f}], difference: {abs(mean_0 - mean_1):.4f}")
        self._log(f"Adjusted means: [{result[s == 0].mean():.4f}, {result[s == 1].mean():.4f}], difference: {abs(result[s == 0].mean() - result[s == 1].mean()):.4f}")
        
        return result
    
    
    def _adjust_for_equal_opportunity(self, baseline_probs: np.ndarray, 
                                    cf_probs: np.ndarray, 
                                    s: np.ndarray) -> np.ndarray:
        """
        Adjust predictions to satisfy equal opportunity without access to true labels.
        Uses the group statistics learned during training to guide adjustments.
        
        Args:
            baseline_probs: Original prediction probabilities
            cf_probs: Counterfactual prediction probabilities
            s: Protected attribute values
            
        Returns:
            Adjusted fair probabilities
        """
        self._log("Adjusting for equal opportunity...")
        
        # Check if we have the necessary statistics
        if 'tpr_0' not in self.group_statistics or 'tpr_1' not in self.group_statistics:
            self._log("Missing TPR statistics. Falling back to demographic parity adjustment.", level="warning")
            return self._adjust_for_demographic_parity(baseline_probs, cf_probs, s)
        
        # Get TPR statistics from training
        tpr_0 = self.group_statistics['tpr_0']
        tpr_1 = self.group_statistics['tpr_1']
        tpr_diff = abs(tpr_0 - tpr_1)
        
        # Get positive class mean and std for each group
        if 'pos_mean_0' not in self.group_statistics or 'pos_mean_1' not in self.group_statistics:
            self._log("Missing positive class statistics. Falling back to demographic parity adjustment.", level="warning")
            return self._adjust_for_demographic_parity(baseline_probs, cf_probs, s)
        
        pos_mean_0 = self.group_statistics['pos_mean_0']
        pos_mean_1 = self.group_statistics['pos_mean_1']
        pos_std_0 = self.group_statistics['pos_std_0']
        pos_std_1 = self.group_statistics['pos_std_1']
        
        # Calculate the target TPR (weighted average based on adjustment strength)
        target_tpr = (tpr_0 + tpr_1) / 2
        
        # Identify likely positive examples based on similarity to the positive class distribution
        # We use a probabilistic approach rather than requiring true labels
        
        # For each group, identify examples that are likely to be positive
        result = baseline_probs.copy()
        
        # Function to identify likely positive examples and adjust them
        def adjust_group_predictions(group_idx, target_tpr, curr_tpr):
            # Get mask for this group
            group_mask = (s == group_idx)
            group_probs = baseline_probs[group_mask]
            group_cf_probs = cf_probs[group_mask]
            
            if len(group_probs) == 0:
                return  # Skip if no examples for this group
            
            # Get group statistics
            group_pos_mean = self.group_statistics[f'pos_mean_{group_idx}']
            group_pos_std = self.group_statistics[f'pos_std_{group_idx}']
            
            # Calculate probability of being positive for each example
            # Using a simple heuristic based on distance from positive class mean
            z_scores = (group_probs - group_pos_mean) / group_pos_std
            pos_probability = np.exp(-0.5 * z_scores**2)  # Approximation based on normal distribution
            pos_probability = pos_probability / pos_probability.max()  # Normalize
            
            # Sort examples by likelihood of being positive
            sorted_indices = np.argsort(-pos_probability)  # Descending order
            
            # How many examples to adjust
            # We estimate this based on the total expected positives
            n_examples = len(group_probs)
            expected_positives = int(n_examples * max(tpr_0, tpr_1))  # Conservative estimate
            
            # Calculate adjustment strength based on how far current TPR is from target
            # tpr_adjustment = self.adjustment_strength * (target_tpr - curr_tpr) * 10
            sign = np.sign(target_tpr - curr_tpr)
            
            # Apply adjustments to the most likely positive examples
            for idx in sorted_indices[:expected_positives]:
                # Greater adjustment for examples more likely to be positive
                likelihood_factor = pos_probability[idx]
                alpha = max(0, min(1, 0.5 - sign * self.adjustment_strength * likelihood_factor))#tpr_adjustment
                
                # Get original index in the full dataset
                original_idx = np.where(group_mask)[0][idx]
                
                # Apply adjustment
                result[original_idx] = alpha * group_probs[idx] + (1 - alpha) * group_cf_probs[idx]
        
        # Adjust both groups
        adjust_group_predictions(0, target_tpr, tpr_0)
        adjust_group_predictions(1, target_tpr, tpr_1)
        
        # Ensure probabilities are in [0, 1]
        result = np.clip(result, 0, 1)
        
        # Log the adjustment results (just the means, as we can't calculate actual TPRs without true labels)
        self._log(f"EO adjustment - Original TPRs: [{tpr_0:.4f}, {tpr_1:.4f}], difference: {tpr_diff:.4f}")
        self._log(f"Adjusted group means: [{result[s == 0].mean():.4f}, {result[s == 1].mean():.4f}]")
        
        return result
    

    def _adjust_for_equalized_odds(self, baseline_probs: np.ndarray, 
                                 cf_probs: np.ndarray, 
                                 s: np.ndarray) -> np.ndarray:
        """
        Adjust predictions to satisfy equalized odds without access to true labels.
        Uses the group statistics learned during training to guide adjustments.
        
        Args:
            baseline_probs: Original prediction probabilities
            cf_probs: Counterfactual prediction probabilities
            s: Protected attribute values
            
        Returns:
            Adjusted fair probabilities
        """
        self._log("Adjusting for equalized odds...")
        
        # Check if we have the necessary statistics
        required_stats = ['tpr_0', 'tpr_1', 'fpr_0', 'fpr_1', 
                         'pos_mean_0', 'pos_mean_1', 'pos_std_0', 'pos_std_1',
                         'neg_mean_0', 'neg_mean_1', 'neg_std_0', 'neg_std_1']
        
        if not all(stat in self.group_statistics for stat in required_stats):
            self._log("Missing required statistics. Falling back to demographic parity adjustment.", level="warning")
            return self._adjust_for_demographic_parity(baseline_probs, cf_probs, s)
        
        # Get statistics from training
        tpr_0 = self.group_statistics['tpr_0']
        tpr_1 = self.group_statistics['tpr_1']
        fpr_0 = self.group_statistics['fpr_0']
        fpr_1 = self.group_statistics['fpr_1']
        
        # Calculate differences
        tpr_diff = abs(tpr_0 - tpr_1)
        fpr_diff = abs(fpr_0 - fpr_1)
        
        # Calculate target rates (averages)
        target_tpr = (tpr_0 + tpr_1) / 2
        target_fpr = (fpr_0 + fpr_1) / 2
        
        # Create result array
        result = baseline_probs.copy()
        
        # Function to identify likely positive/negative examples and adjust them
        def adjust_group_predictions(group_idx, target_tpr, curr_tpr, target_fpr, curr_fpr):
            # Get mask for this group
            group_mask = (s == group_idx)
            group_probs = baseline_probs[group_mask]
            group_cf_probs = cf_probs[group_mask]
            
            if len(group_probs) == 0:
                return  # Skip if no examples for this group
            
            # Get group statistics
            group_pos_mean = self.group_statistics[f'pos_mean_{group_idx}']
            group_pos_std = self.group_statistics[f'pos_std_{group_idx}']
            group_neg_mean = self.group_statistics[f'neg_mean_{group_idx}']
            group_neg_std = self.group_statistics[f'neg_std_{group_idx}']
            
            # Calculate probability of being positive for each example
            z_scores_pos = (group_probs - group_pos_mean) / group_pos_std
            pos_probability = np.exp(-0.5 * z_scores_pos**2)
            pos_probability = pos_probability / pos_probability.max()  # Normalize
            
            # Calculate probability of being negative for each example
            z_scores_neg = (group_probs - group_neg_mean) / group_neg_std
            neg_probability = np.exp(-0.5 * z_scores_neg**2)
            neg_probability = neg_probability / neg_probability.max()  # Normalize
            
            # Normalize to make them pseudo-probabilities
            total_prob = pos_probability + neg_probability
            pos_probability = pos_probability / total_prob
            neg_probability = neg_probability / total_prob
            
            # Get original indices in the full dataset
            original_indices = np.where(group_mask)[0]
            
            # Calculate adjustments for TPR and FPR
            tpr_adjustment = self.adjustment_strength * (target_tpr - curr_tpr)
            sign_tpr = np.sign(target_tpr - curr_tpr)
            fpr_adjustment = self.adjustment_strength * (target_fpr - curr_fpr)
            sign_fpr = np.sign(target_fpr - curr_fpr)
            
            # Apply adjustments based on likelihood of being positive or negative
            for i, (pos_prob, neg_prob, original_idx) in enumerate(zip(pos_probability, neg_probability, original_indices)):
                # Determine which adjustment to apply based on which class is more likely
                if pos_prob > neg_prob:
                    # More likely to be positive, apply TPR adjustment
                    alpha = max(0, min(1, 0.5 - sign_tpr * self.adjustment_strength * pos_prob))#tpr_adjustment
                else:
                    # More likely to be negative, apply FPR adjustment
                    alpha = max(0, min(1, 0.5 - sign_fpr * self.adjustment_strength * neg_prob))#fpr_adjustment
                
                # Apply adjustment
                result[original_idx] = alpha * group_probs[i] + (1 - alpha) * group_cf_probs[i]
        
        # Adjust both groups
        adjust_group_predictions(0, target_tpr, tpr_0, target_fpr, fpr_0)
        adjust_group_predictions(1, target_tpr, tpr_1, target_fpr, fpr_1)
        
        # Ensure probabilities are in [0, 1]
        result = np.clip(result, 0, 1)
        
        # Log the adjustment results
        self._log(f"EOd adjustment - Original TPRs: [{tpr_0:.4f}, {tpr_1:.4f}], difference: {tpr_diff:.4f}")
        self._log(f"Original FPRs: [{fpr_0:.4f}, {fpr_1:.4f}], difference: {fpr_diff:.4f}")
        self._log(f"Adjusted group means: [{result[s == 0].mean():.4f}, {result[s == 1].mean():.4f}]")
        
        return result
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, s: np.ndarray, 
                output_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance and fairness.
        
        Args:
            X: Feature matrix
            y: Target variable
            s: Protected attribute values
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with evaluation results
        """
        import fairness_metrics as fairness
        
        if self.base_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get baseline predictions
        baseline_preds = self.base_model.predict(X)
        baseline_probs = self.base_model.predict_proba(X)[:, 1]
        
        # Get fair predictions
        fair_preds = self.predict(X, s)
        fair_probs = self.predict_proba(X, s)[:, 1]
        
        # Performance metrics
        baseline_performance = {
            'accuracy': accuracy_score(y, baseline_preds),
            'f1_score': f1_score(y, baseline_preds),
            'roc_auc': roc_auc_score(y, baseline_probs)
        }
        
        fair_performance = {
            'accuracy': accuracy_score(y, fair_preds),
            'f1_score': f1_score(y, fair_preds),
            'roc_auc': roc_auc_score(y, fair_probs)
        }
        
        # Fairness metrics
        baseline_fairness_report = fairness.fairness_report(
            y, baseline_preds, baseline_probs, s, group_names=['Group 0', 'Group 1']
        )
        
        fair_fairness_report = fairness.fairness_report(
            y, fair_preds, fair_probs, s, group_names=['Group 0', 'Group 1']
        )
        
        baseline_fairness = baseline_fairness_report['fairness']
        fair_fairness = fair_fairness_report['fairness']
        
        # Calculate improvements
        fairness_improvement = {}
        for metric in baseline_fairness:
            if metric in fair_fairness:
                if metric == 'disparate_impact_ratio':
                    # For disparate impact, 1.0 is optimal
                    baseline_val = abs(baseline_fairness[metric] - 1.0)
                    fair_val = abs(fair_fairness[metric] - 1.0)
                    if baseline_val > 0:
                        improvement = (baseline_val - fair_val) / baseline_val * 100
                    else:
                        improvement = 0.0
                else:
                    # For other metrics, lower is better
                    baseline_val = baseline_fairness[metric]
                    fair_val = fair_fairness[metric]
                    if baseline_val > 0:
                        improvement = (baseline_val - fair_val) / baseline_val * 100
                    else:
                        improvement = 0.0
                
                fairness_improvement[metric] = improvement
        
        # Combine all metrics
        results = {
            'baseline': {
                'performance': baseline_performance,
                'fairness': baseline_fairness
            },
            'fair': {
                'performance': fair_performance,
                'fairness': fair_fairness
            },
            'improvement': fairness_improvement
        }
        
        # Print summary
        logger.info("=== Evaluation Results ===")
        logger.info(f"Baseline Accuracy: {baseline_performance['accuracy']:.4f}")
        logger.info(f"Fair Model Accuracy: {fair_performance['accuracy']:.4f}")
        logger.info(f"Accuracy Change: {fair_performance['accuracy'] - baseline_performance['accuracy']:.4f}")
        
        for metric in fairness_improvement:
            baseline_val = baseline_fairness[metric]
            fair_val = fair_fairness[metric]
            improvement = fairness_improvement[metric]
            logger.info(f"{metric}: {baseline_val:.4f} -> {fair_val:.4f} (Improvement: {improvement:.1f}%)")
        
        # Generate visualizations if output directory provided
        if output_dir:
            try:
                import visualization as viz
                import matplotlib.pyplot as plt
                import os
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Plot fairness metrics comparison
                fairness_path = os.path.join(output_dir, "fairness_comparison.png")
                comparison_fig = viz.plot_fairness_metrics_comparison(baseline_fairness, fair_fairness, save_path=fairness_path)
                if comparison_fig is not None:
                    plt.close(comparison_fig)
                
                # Plot outcome probability distributions
                probs_path = os.path.join(output_dir, "outcome_probabilities.png")
                probs_fig = viz.plot_outcome_probabilities(baseline_probs, fair_probs, s, save_path=probs_path)
                if probs_fig is not None:
                    plt.close(probs_fig)
                
                # Generate and plot counterfactuals if possible
                try:
                    X_cf = self.counterfactual_generator.transform(X, s, self.feature_names)
                    
                    # Plot counterfactual distributions
                    cf_dist_path = os.path.join(output_dir, "counterfactual_distributions.png")
                    cf_dist_fig = viz.plot_counterfactual_distributions(X, X_cf, s, self.feature_names, save_path=cf_dist_path)
                    if cf_dist_fig is not None:
                        plt.close(cf_dist_fig)
                    
                    # Plot embedding space
                    embedding_path = os.path.join(output_dir, "embedding_space.png")
                    embedding_fig = viz.plot_embedding_space(X, X_cf, s, y, save_path=embedding_path)
                    if embedding_fig is not None:
                        plt.close(embedding_fig)
                except Exception as e:
                    logger.error(f"Error generating counterfactual visualizations: {e}")
                
                # Plot ROC curves
                try:
                    baseline_roc_path = os.path.join(output_dir, "baseline_roc_curves.png")
                    baseline_fig = fairness.plot_roc_curves(y, baseline_probs, s, ['Group 0', 'Group 1'], 
                                                "Baseline Model ROC Curves", save_path=baseline_roc_path)
                    if baseline_fig is not None:
                        plt.close(baseline_fig)
                    
                    fair_roc_path = os.path.join(output_dir, "fair_roc_curves.png")
                    fair_fig = fairness.plot_roc_curves(y, fair_probs, s, ['Group 0', 'Group 1'], 
                                            "Fair Model ROC Curves", save_path=fair_roc_path)
                    if fair_fig is not None:
                        plt.close(fair_fig)
                    
                    # Plot combined ROC curves if function exists
                    if hasattr(viz, 'plot_roc_curves_combined'):
                        combined_roc_path = os.path.join(output_dir, "combined_roc_curves.png")
                        combined_fig = viz.plot_roc_curves_combined(y, baseline_probs, fair_probs, s, 
                                                               ['Group 0', 'Group 1'], save_path=combined_roc_path)
                        if combined_fig is not None:
                            plt.close(combined_fig)
                except Exception as e:
                    logger.error(f"Error generating ROC curve visualizations: {e}")
                
                # Feature importance plot if available
                if hasattr(self.base_model, 'feature_importances_') and self.base_model.feature_importances_ is not None:
                    try:
                        importance_path = os.path.join(output_dir, "feature_importances.png")
                        viz.plot_feature_importance(self.base_model.feature_importances_, 
                                                  self.feature_names, save_path=importance_path)
                    except Exception as e:
                        logger.error(f"Error generating feature importance plot: {e}")
            
            except Exception as e:
                logger.error(f"Error in visualization generation: {e}")
        
        return results
    
    def save(self, model_dir: str) -> None:
        """
        Save the model to disk.
        
        Args:
            model_dir: Directory to save model components
        """
        import os
        import json
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save base model
        self.base_model.save_model(os.path.join(model_dir, "base_model.joblib"))
        
        # Save causal graph
        self.causal_graph.save(os.path.join(model_dir, "causal_graph.pkl"))
        
        # Save counterfactual generator
        self.counterfactual_generator.save(os.path.join(model_dir, "counterfactual_generator.pkl"))
        
        # Save model configuration and statistics
        with open(os.path.join(model_dir, "model_config.json"), 'w') as f:
            # Convert numpy values to native Python types for JSON serialization
            def convert_value(v):
                if isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(v)
                elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                    return float(v)
                elif isinstance(v, (np.ndarray,)):
                    return v.tolist()
                else:
                    return v
            
            # Convert dictionaries with numpy values
            fairness_statistics = {k: convert_value(v) for k, v in self.fairness_statistics.items()}
            group_statistics = {k: convert_value(v) for k, v in self.group_statistics.items()}
            baseline_metrics = {k: convert_value(v) for k, v in self.baseline_metrics.items()}
            
            json.dump({
                'feature_names': self.feature_names,
                'base_model_type': self.base_model_type,
                'base_model_params': self.base_model_params,
                'counterfactual_method': self.counterfactual_method,
                'adjustment_strength': self.adjustment_strength,
                'fairness_constraint': self.fairness_constraint,
                'fairness_threshold': self.fairness_threshold,
                'amplification_factor': self.amplification_factor,
                'random_state': self.random_state,
                'fairness_weight': self.fairness_weight,
                'prediction_threshold': self.prediction_threshold,
                'verbose': self.verbose,
                'fairness_statistics': fairness_statistics,
                'group_statistics': group_statistics,
                'baseline_metrics': baseline_metrics
            }, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str) -> 'TSCFM':
        """
        Load model from disk.
        
        Args:
            model_dir: Directory with saved model components
            
        Returns:
            Loaded TSCFM model
        """
        import os
        import json
        from base_models import BaseModelWrapper
        from causal_graph import CausalGraph
        from counterfactual_generator import CounterfactualGenerator
        
        # Load configuration
        with open(os.path.join(model_dir, "model_config.json"), 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            base_model_type=config['base_model_type'],
            base_model_params=config['base_model_params'],
            counterfactual_method=config['counterfactual_method'],
            adjustment_strength=config['adjustment_strength'],
            fairness_constraint=config['fairness_constraint'],
            fairness_threshold=config['fairness_threshold'],
            amplification_factor=config.get('amplification_factor', 1.5),
            random_state=config['random_state'],
            fairness_weight=config.get('fairness_weight', 0.7),
            prediction_threshold=config.get('prediction_threshold', 0.5),
            verbose=config.get('verbose', False)
        )
        
        # Load saved state
        model.feature_names = config['feature_names']
        model.fairness_statistics = config['fairness_statistics']
        model.group_statistics = config['group_statistics']
        model.baseline_metrics = config['baseline_metrics']
        
        # Load base model
        model.base_model = BaseModelWrapper.load_model(os.path.join(model_dir, "base_model.joblib"))
        
        # Load causal graph
        model.causal_graph = CausalGraph.load(os.path.join(model_dir, "causal_graph.pkl"))
        
        # Load counterfactual generator
        model.counterfactual_generator = CounterfactualGenerator.load(
            os.path.join(model_dir, "counterfactual_generator.pkl"),
            causal_graph=model.causal_graph
        )
        
        logger.info(f"Model loaded from {model_dir}")
        return model
