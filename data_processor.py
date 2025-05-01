import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional, Union
import matplotlib.pyplot as plt

# Import the dataset loading functions
from load_german import load_german
from load_card_credit import load_card_credit
from load_pakdd import load_pkdd

class TSCFMDataProcessor:
    """
    Data processor for the Two-Stage Counterfactual Fairness Model.
    Handles dataset loading, preprocessing, and feature categorization.
    """
    
    def __init__(self, dataset_name: str, test_size: float = 0.3, random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            dataset_name: Name of the dataset ('german', 'card_credit', or 'pakdd')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.dataset_name = dataset_name.lower()
        self.test_size = test_size
        self.random_state = random_state
        
        # Will be populated after load_data is called
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.s_train = None
        self.s_test = None
        
        # Will store feature names
        self.feature_names = None
        
        # Will store feature categories
        self.direct_features = []
        self.proxy_features = []
        self.mediator_features = []
        self.neutral_features = []
        
        # Original dataframe if available
        self.df = None
        
        # Track if data is loaded
        self.data_loaded = False
        
    def load_data(self, filepath: Optional[str] = None) -> None:
        """
        Load and preprocess the specified dataset.
        
        Args:
            filepath: Path to the dataset file for card_credit or pakdd
        """
        if self.dataset_name == 'german':
            X, y, s = load_german(filepath, mode='label_encoding')
            # Create feature names since they're not provided by the loader
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        elif self.dataset_name == 'card_credit':
            if filepath is None:
                raise ValueError("Filepath must be provided for card_credit dataset")
            X, y, s = load_card_credit(filepath, mode='label_encoding')
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        elif self.dataset_name == 'pakdd':
            if filepath is None:
                raise ValueError("Filepath must be provided for pakdd dataset")
            X, y, s = load_pkdd(filepath, mode='label_encoding')
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Store the data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.s_train = s_train
        self.s_test = s_test
        
        # Mark data as loaded
        self.data_loaded = True
        
        print(f"Data loaded: {self.dataset_name}")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"Class distribution (train): {np.bincount(y_train.astype(int))}")
        print(f"Gender distribution (train): {np.bincount(s_train.astype(int))}")
    
    def analyze_feature_correlations(self, threshold: float = 0.2) -> Dict[str, List[int]]:
        """
        Analyze correlations between features and sensitive attribute to categorize features.
        
        Args:
            threshold: Correlation threshold to consider a feature as correlated with gender
            
        Returns:
            Dictionary with feature categories (direct, proxy, mediator, neutral)
        """
        if not self.data_loaded:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        # Calculate correlation with sensitive attribute
        X_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        X_df['gender'] = self.s_train
        X_df['target'] = self.y_train
        
        # Correlation with gender
        gender_corr = np.abs(
            [np.corrcoef(X_df[col], X_df['gender'])[0, 1] for col in self.feature_names]
        )
        
        # Correlation with target
        target_corr = np.abs(
            [np.corrcoef(X_df[col], X_df['target'])[0, 1] for col in self.feature_names]
        )
        
        # Initialize feature categories
        self.direct_features = []
        self.proxy_features = []
        self.mediator_features = []
        self.neutral_features = []
        
        # Categorize features based on correlations
        for i, (g_corr, t_corr) in enumerate(zip(gender_corr, target_corr)):
            if g_corr > threshold:
                if t_corr > threshold:
                    self.mediator_features.append(i)  # Affects both gender and target
                else:
                    self.direct_features.append(i)    # Directly related to gender
            else:
                if t_corr > threshold:
                    self.proxy_features.append(i)     # Could be a proxy for gender
                else:
                    self.neutral_features.append(i)   # Not strongly related to either
        
        feature_categories = {
            'direct': self.direct_features,
            'proxy': self.proxy_features,
            'mediator': self.mediator_features,
            'neutral': self.neutral_features
        }
        
        print(f"Feature categories identified:")
        print(f"  Direct features: {len(self.direct_features)}")
        print(f"  Proxy features: {len(self.proxy_features)}")
        print(f"  Mediator features: {len(self.mediator_features)}")
        print(f"  Neutral features: {len(self.neutral_features)}")
        
        return feature_categories
    
    def visualize_correlations(self, save_path: Optional[str] = None) -> None:
        """
        Visualize correlations between features, sensitive attribute, and target.
        
        Args:
            save_path: If provided, save the plot to this path
        """
        if not self.data_loaded:
            raise ValueError("Data must be loaded first. Call load_data()")
            
        X_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        X_df['gender'] = self.s_train
        X_df['target'] = self.y_train
        
        # Calculate correlations
        gender_corr = [np.corrcoef(X_df[col], X_df['gender'])[0, 1] for col in self.feature_names]
        target_corr = [np.corrcoef(X_df[col], X_df['target'])[0, 1] for col in self.feature_names]
        
        # Create scatter plot of correlations
        plt.figure(figsize=(10, 6))
        plt.scatter(gender_corr, target_corr, alpha=0.6)
        
        # Add quadrant lines
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Label axes
        plt.xlabel('Correlation with Gender')
        plt.ylabel('Correlation with Target')
        plt.title(f'Feature Correlations - {self.dataset_name.capitalize()} Dataset')
        
        # Add some feature labels for highly correlated features
        threshold = 0.2
        for i, (g_corr, t_corr) in enumerate(zip(gender_corr, target_corr)):
            if abs(g_corr) > threshold or abs(t_corr) > threshold:
                plt.annotate(self.feature_names[i], (g_corr, t_corr), fontsize=8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        plt.show()
    
    def get_feature_categories(self) -> Dict[str, List[int]]:
        """
        Get the current feature categories.
        
        Returns:
            Dictionary with feature categories
        """
        return {
            'direct': self.direct_features,
            'proxy': self.proxy_features,
            'mediator': self.mediator_features,
            'neutral': self.neutral_features
        }
    
    def select_features_by_category(self, categories: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select features from specified categories.
        
        Args:
            categories: List of category names ('direct', 'proxy', 'mediator', 'neutral')
            
        Returns:
            Tuple of (X_train_selected, X_test_selected)
        """
        if not self.data_loaded:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        # Get indices of features to select
        selected_indices = []
        for category in categories:
            if category == 'direct':
                selected_indices.extend(self.direct_features)
            elif category == 'proxy':
                selected_indices.extend(self.proxy_features)
            elif category == 'mediator':
                selected_indices.extend(self.mediator_features)
            elif category == 'neutral':
                selected_indices.extend(self.neutral_features)
            else:
                raise ValueError(f"Unknown category: {category}")
        
        # Sort indices to maintain original order
        selected_indices = sorted(selected_indices)
        
        # Select features
        X_train_selected = self.X_train[:, selected_indices]
        X_test_selected = self.X_test[:, selected_indices]
        
        return X_train_selected, X_test_selected
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the processed data.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, s_train, s_test)
        """
        if not self.data_loaded:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.s_train, self.s_test
    
    def get_data_by_gender(self) -> Dict[str, np.ndarray]:
        """
        Split the data by gender for counterfactual analysis.
        
        Returns:
            Dictionary with data split by gender
        """
        if not self.data_loaded:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        # Split training data by gender
        male_mask = self.s_train == 0
        female_mask = self.s_train == 1
        
        X_train_male = self.X_train[male_mask]
        X_train_female = self.X_train[female_mask]
        y_train_male = self.y_train[male_mask]
        y_train_female = self.y_train[female_mask]
        
        return {
            'X_train_male': X_train_male,
            'X_train_female': X_train_female,
            'y_train_male': y_train_male,
            'y_train_female': y_train_female
        }
