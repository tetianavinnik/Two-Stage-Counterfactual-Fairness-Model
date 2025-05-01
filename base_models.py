"""
Implementation of base models for Stage 1 of the Two-Stage Counterfactual Fairness Model.
This module provides wrappers around different classifier types with standardized interfaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os


class BaseModelWrapper:
    """
    Base wrapper for Stage 1 models in TSCFM.
    
    This class provides a standardized interface for different classifier types
    and includes methods for hyperparameter optimization.
    """
    
    def __init__(self, model_type: str = "random_forest", 
                 hyperparams: Optional[Dict[str, Any]] = None,
                 random_state: int = 42):
        """
        Initialize the base model wrapper.
        
        Args:
            model_type: Type of model to use ('random_forest', 'logistic_regression', 
                       'gradient_boosting', or 'neural_network')
            hyperparams: Hyperparameters for the model
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.hyperparams = hyperparams if hyperparams is not None else {}
        self.random_state = random_state
        self.model = self._create_model()
        self.feature_importances_ = None
    
    def _create_model(self) -> BaseEstimator:
        """
        Create a model based on the specified type.
        
        Returns:
            A scikit-learn estimator
        """
        hyperparams = self.hyperparams.copy()
        hyperparams['random_state'] = self.random_state
        
        if self.model_type == "random_forest":
            return RandomForestClassifier(**hyperparams)
        
        elif self.model_type == "logistic_regression":
            return LogisticRegression(**hyperparams)
        
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(**hyperparams)
        
        elif self.model_type == "neural_network":
            return MLPClassifier(**hyperparams)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModelWrapper':
        """
        Fit the model to the data.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(self.model.coef_.shape) > 1:
                self.feature_importances_ = np.abs(self.model.coef_[0])
            else:
                self.feature_importances_ = np.abs(self.model.coef_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities for each class
        """
        return self.model.predict_proba(X)
    
    def optimize_hyperparams(self, X: np.ndarray, y: np.ndarray, 
                           param_grid: Dict[str, List[Any]],
                           cv: int = 5, n_iter: Optional[int] = None,
                           scoring: str = 'roc_auc',
                           n_jobs: int = -1) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters using cross-validation.
        
        Args:
            X: Features
            y: Target variable
            param_grid: Grid of hyperparameters to search
            cv: Number of cross-validation folds
            n_iter: Number of iterations for randomized search (if None, use grid search)
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best_params, best_score)
        """
        # Create a new model instance
        model = self._create_model()
        
        # Determine search strategy based on n_iter
        if n_iter is None:
            # Use grid search
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        else:
            # Use randomized search
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, 
                                       scoring=scoring, n_jobs=n_jobs, random_state=self.random_state)
        
        # Fit the search
        search.fit(X, y)
        
        # Update hyperparams and model
        self.hyperparams.update(search.best_params_)
        self.model = self._create_model()
        self.model.fit(X, y)
        
        # Update feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(self.model.coef_.shape) > 1:
                self.feature_importances_ = np.abs(self.model.coef_[0])
            else:
                self.feature_importances_ = np.abs(self.model.coef_)
        
        return search.best_params_, search.best_score_
    
    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> pd.Series:
        """
        Get feature importances as a pandas Series.
        
        Args:
            feature_names: Names of features (if None, use indices)
            
        Returns:
            Pandas Series with feature importances
        """
        if self.feature_importances_ is None:
            raise ValueError("Model does not have feature importances. Call fit() first.")
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(self.feature_importances_))]
        
        return pd.Series(self.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseModelWrapper':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        return joblib.load(filepath)


# Dictionary of default hyperparameter grids for optimization
DEFAULT_PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    },
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "class_weight": [None, "balanced"]
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_split": [2, 5, 10]
    },
    "neural_network": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
        "solver": ["adam", "sgd"],
        "max_iter": [200, 500]
    }
}


def get_default_param_grid(model_type: str) -> Dict[str, List[Any]]:
    """
    Get the default hyperparameter grid for a model type.
    
    Args:
        model_type: Type of model ('random_forest', 'logistic_regression', 
                   'gradient_boosting', or 'neural_network')
        
    Returns:
        Hyperparameter grid for optimization
    """
    model_type = model_type.lower()
    if model_type not in DEFAULT_PARAM_GRIDS:
        raise ValueError(f"No default parameter grid for model type: {model_type}")
    
    return DEFAULT_PARAM_GRIDS[model_type]
