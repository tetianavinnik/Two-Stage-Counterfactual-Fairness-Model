import os
import pandas as pd
import numpy as np
from datetime import datetime

# Constants for experiment setup
DATASETS = ["german", "card_credit", "pakdd"]
COUNTERFACTUAL_METHODS = ["structural_equation", "matching", "generative"]
ADJUSTMENT_STRENGTHS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
AMPLIFICATION_FACTORS = [0.5, 1.0, 2.0, 3.0]

FAIRNESS_CONSTRAINTS = [
    "demographic_parity_difference",
    "equal_opportunity_difference", 
    "equalized_odds_difference"
]

# Directory setup
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = "./results/tscfm_experiments"
DEFAULT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"experiment_{TIMESTAMP}")

# Random seed for reproducibility
RANDOM_SEED = 42

# Benchmark results from previous study
BENCHMARK_RESULTS = {
    "german": pd.DataFrame({
        "Model": ["DT", "NB", "MLP", "kNN", "DIR-DT", "DIR-NB", "DIR-MLP", "DIR-kNN", 
                  "LFR-DT", "LFR-NB", "LFR-MLP", "LFR-kNN", "AdaFair", "Agarwal's", 
                  "EOP-DT", "EOP-NB", "EOP-MLP", "EOP-kNN", "CEP-DT", "CEP-NB", "CEP-MLP", "CEP-kNN"],
        "BA": [0.5954, 0.6604, 0.6095, 0.5348, 0.6221, 0.6392, 0.5676, 0.5118, 
               0.5686, 0.4861, 0.5, 0.5467, 0.5, 0.6289, 0.5954, 0.6286, 0.5990, 
               0.5309, 0.5954, 0.6153, 0.5667, 0.5257],
        "Acc": [0.6567, 0.7300, 0.6634, 0.6500, 0.6767, 0.7133, 0.7000, 0.6267, 
                0.5933, 0.5433, 0.6967, 0.6667, 0.6967, 0.7033, 0.6567, 0.6900, 
                0.6533, 0.6533, 0.6567, 0.7233, 0.6600, 0.6633],
        "SP": [0.0485, 0.0019, -0.0669, 0.0641, -0.0736, -0.0094, -0.0326, -0.0144, 
               -0.0174, -0.0592, 0.0, -0.0570, 0.0, -0.0945, 0.0485, -0.0935, 
               -0.0877, 0.0877, 0.0485, 0.1151, 0.0510, 0.1396],
        "EO": [0.0160, 0.0614, 0.0936, 0.0670, 0.0972, 0.0511, 0.0625, 0.0608, 
               0.0646, 0.0410, 0.0, 0.0649, 0.0, 0.1116, 0.0160, 0.1347, 0.1069, 
               0.0870, 0.0160, 0.0120, 0.0136, 0.1337],
        "EOd": [0.1807, 0.1615, 0.1292, 0.1171, 0.1489, 0.0983, 0.0781, 0.1431, 
                0.1325, 0.1361, 0.0, 0.1161, 0.0, 0.2001, 0.1807, 0.1703, 0.1770, 
                0.1693, 0.1807, 0.3218, 0.1877, 0.2805],
        "PP": [0.0292, 0.0166, 0.0214, 0.0391, 0.0263, 0.0043, 0.0169, 0.0018, 
               0.0128, 0.0546, 0.0371, 0.0336, 0.0371, 0.0350, 0.0292, 0.0131, 
               0.0297, 0.0357, 0.0292, 0.0597, 0.0227, 0.0307],
        "PE": [0.1646, 0.1001, 0.0356, 0.0501, 0.0517, 0.0473, 0.0156, 0.0823, 
               0.0679, 0.095, 0.0, 0.0512, 0.0, 0.0884, 0.1646, 0.0356, 0.0701, 
               0.0823, 0.1646, 0.3098, 0.1741, 0.1468],
        "TE": [0.0769, -0.2557, -0.6250, 0.1399, -0.6653, -0.2667, -0.2178, -0.2114, 
               -0.3510, -0.5092, 0.0, -0.2826, 0.0, -0.6364, 0.0769, -0.7500, 
               -0.7279, 0.2190, 0.0769, 0.1980, 0.0208, 0.3783],
        "ABROCA": [0.0903, 0.1012, 0.0697, 0.0458, 0.0227, 0.0970, 0.1114, 0.1223, 
                   0.0032, 0.0342, 0.0545, 0.0434, 0.0534, 0.0384, float('nan'), 
                   float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 
                   float('nan'), float('nan')]
    }),

    "card_credit": pd.DataFrame({
        "Model": ["DT", "NB", "MLP", "kNN", "DIR-DT", "DIR-NB", "DIR-MLP", "DIR-kNN", 
                "LFR-DT", "LFR-NB", "LFR-MLP", "LFR-kNN", "AdaFair", "Agarwal's", 
                "EOP-DT", "EOP-NB", "EOP-MLP", "EOP-kNN", "CEP-DT", "CEP-NB", "CEP-MLP", "CEP-kNN"],
        "BA": [0.6131, 0.5599, 0.6111, 0.5435, 0.6099, 0.5674, 0.5301, 0.5471, 
            0.5798, 0.4831, 0.4514, 0.4967, 0.6392, 0.5625, 0.6132, 0.5548, 0.6073, 
            0.5416, 0.6131, 0.5599, 0.6111, 0.5407],
        "Acc": [0.7277, 0.3778, 0.5782, 0.7530, 0.7187, 0.4404, 0.7814, 0.7511, 
                0.5897, 0.7103, 0.6406, 0.2270, 0.8200, 0.5270, 0.7278, 0.3714, 
                0.5812, 0.7534, 0.7277, 0.3778, 0.5782, 0.7561],
        "SP": [0.0308, -0.0034, 0.0523, 0.0153, 0.0290, 0.0121, 0.0245, 0.0053, 
            0.0476, -0.0051, 0.0117, -0.0041, 0.0045, 0.0045, 0.0308, 0.0090, 
            0.0138, 0.0062, 0.0308, -0.0034, 0.0522, 0.0392],
        "EO": [0.0263, 0.0308, 0.0403, 0.0089, 0.0034, 0.0174, 0.0258, 0.0106, 
            0.0039, 0.0081, 0.0264, 0.0051, 0.0342, 0.0228, 0.0263, 0.0379, 
            0.0026, 0.0053, 0.0263, 0.0308, 0.0403, 0.0322],
        "EOd": [0.0656, 0.0311, 0.0883, 0.0230, 0.0342, 0.0333, 0.0479, 0.0111, 
                0.0586, 0.0161, 0.0373, 0.0091, 0.0402, 0.0238, 0.0652, 0.0566, 0.0137, 
                0.0119, 0.0656, 0.0311, 0.0883, 0.0590],
        "PP": [0.0275, 0.0226, 0.0231, 0.0105, 0.0011, 0.0215, 0.0122, 0.0482, 
            0.0061, 0.0457, 0.0384, 0.0267, 0.0207, 0.0362, 0.0271, 0.0165, 
            0.0267, 0.0107, 0.0275, 0.0226, 0.0231, 0.0051],
        "PE": [0.0393, 0.0002, 0.0481, 0.0142, 0.0308, 0.0158, 0.0220, 0.0005, 
            0.0546, 0.0079, 0.0109, 0.0040, 0.0059, 0.0009, 0.0389, 0.0187, 0.0110, 
            0.0066, 0.0393, 0.0002, 0.0481, 0.0268],
        "TE": [0.0071, -0.0184, 0.0158, 0.0454, -0.0079, -0.0152, 7.9436, -0.3479, 
            -0.0043, -0.5313, -0.8892, -0.0026, -0.2942, -0.0392, 0.0056, -0.0199, 
            -0.0311, -0.2388, 0.0071, -0.0184, 0.0148, 0.0475],
        "ABROCA": [0.0324, 0.0238, 0.0193, 0.0115, 0.0174, 0.0240, 0.0129, 0.0101, 
                0.0062, 0.0063, 0.0074, 0.0106, 0.0202, 0.0098, float('nan'), 
                float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 
                float('nan'), float('nan')]
    }),
    
    "pakdd": pd.DataFrame({
        "Model": ["DT", "NB", "MLP", "kNN", "DIR-DT", "DIR-NB", "DIR-MLP", "DIR-kNN", 
                 "LFR-DT", "LFR-NB", "LFR-MLP", "LFR-kNN", "AdaFair", "Agarwal's", 
                 "EOP-DT", "EOP-NB", "EOP-MLP", "EOP-kNN", "CEP-DT", "CEP-NB", "CEP-MLP", "CEP-kNN"],
        "BA": [0.5241, 0.5088, 0.5119, 0.5057, 0.5474, 0.5130, 0.5003, 0.5027, 
               0.4949, 0.5034, 0.4986, 0.5075, 0.5, 0.5093, 0.5241, 0.5083, 0.5132, 0.5057, 
               0.5241, 0.5088, 0.5127, 0.5060],
        "Acc": [0.6244, 0.7256, 0.6925, 0.6822, 0.6189, 0.7210, 0.7351, 0.6810, 
                0.7200, 0.7771, 0.7180, 0.4736, 0.7353, 0.7263, 0.6244, 0.7258, 0.6854, 0.6817, 
                0.6244, 0.7269, 0.7029, 0.6860],
        "SP": [0.0124, 0.0022, 0.0655, -0.0056, 0.0253, 0.0251, 0.0, -0.0028, 
               0.0060, -0.0088, 0.0034, 0.0212, 0.0, -0.0017, 0.0124, -0.0009, 0.0120, -0.0076, 
               0.0124, 0.0068, 0.0996, 0.0069],
        "EO": [0.0325, 0.0087, 0.0715, 0.0192, 0.0116, 0.0156, 0.0018, 0.0044, 
               0.0018, 0.0064, 0.0070, 0.0024, 0.0, 0.0140, 0.0325, 0.0134, 0.0123, 0.0214, 
               0.0325, 0.0037, 0.1051, 0.0070],
        "EOd": [0.0358, 0.0143, 0.1340, 0.0201, 0.0493, 0.0432, 0.0024, 0.0101, 
                0.0100, 0.0163, 0.0145, 0.0319, 0.0, 0.0162, 0.0358, 0.0166, 0.0234, 0.0244, 
                0.0358, 0.0138, 0.2018, 0.0186],
        "PP": [0.0476, 0.0523, 0.0224, 0.0013, 0.0090, 0.0566, 0.5833, 0.0437, 
               0.0189, 0.0271, 0.0543, 0.0155, 0.0, 0.0549, 0.0476, 0.0613, 0.0270, 0.0013, 
               0.0476, 0.0696, 0.0069, 0.0053],
        "PE": [0.0033, 0.0056, 0.0624, 0.0010, 0.0377, 0.0277, 0.0006, 0.0056, 
               0.0093, 0.0100, 0.0074, 0.0294, 0.0, 0.0021, 0.0033, 0.0031, 0.0010, 0.0029, 
               0.0033, 0.0101, 0.0967, 0.0116],
        "TE": [-0.0707, 0.3997, 1.5546, -0.4404, -0.0109, 3.4949, -909.4, -0.4816, 
               2.6914, -0.0033, 0.6436, -0.0239, float('nan'), -0.9031, -0.0707, -0.5122, -0.0755, -0.4848, 
               -0.0707, 2.6232, 4.6288, -0.1113],
        "ABROCA": [0.0146, 0.0110, 0.0144, 0.0094, 0.0247, 0.0112, 0.0128, 0.0134, 
                   0.0040, 0.0017, 0.0072, 0.0159, 0.0141, 0.0081, float('nan'), float('nan'), float('nan'), float('nan'), 
                   float('nan'), float('nan'), float('nan'), float('nan')]
    })
}

# Metric mapping between fairness metrics and column names
METRIC_MAPPING = {
    "balanced_accuracy": "BA",
    "accuracy": "Acc",
    "statistical_parity_difference": "SP",
    "equal_opportunity_difference": "EO",
    "equalized_odds_difference": "EOd",
    "predictive_parity_difference": "PP",
    "predictive_equality_difference": "PE",
    "treatment_equality_difference": "TE",
    "abroca": "ABROCA"
}

# Metric display names (for plots)
METRIC_DISPLAY_NAMES = {
    "BA": "Balanced Accuracy",
    "Acc": "Accuracy",
    "SP": "Statistical Parity Difference",
    "EO": "Equal Opportunity Difference",
    "EOd": "Equalized Odds Difference",
    "PP": "Predictive Parity Difference",
    "PE": "Predictive Equality Difference",
    "TE": "Treatment Equality Difference",
    "ABROCA": "ABROCA"
}

# Full paths to datasets
DATASET_PATHS = {
    "german": r"path_to_file", # Update with your actual path
    "card_credit": r"path_to_file",  # Update with your actual path
    "pakdd": r"path_to_file"  # Update with your actual path
}

# Fairness metrics to focus on
PRIMARY_FAIRNESS_METRICS = ["SP", "EO", "EOd"]
SECONDARY_FAIRNESS_METRICS = ["PP", "PE", "TE", "ABROCA"]

# TSCFM base configuration that will be modified for each experiment
BASE_TSCFM_CONFIG = {
    "base_model_type": "random_forest",
    "base_model_params": {"n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 5,
        "class_weight": "balanced"},
    "random_state": RANDOM_SEED
}

# BASE_TSCFM_CONFIG = {
#     "base_model_type": "random_forest",
#     "base_model_params": {
#         "n_estimators": 300,
#         "max_depth": 20,
#         "min_samples_split": 10,
#         "min_samples_leaf": 5,
#         "max_features": "sqrt",
#         "random_state": 42,
#         "n_jobs": -1
#     },
#     "random_state": RANDOM_SEED
# }

def get_experiment_grid():
    """Generate a grid of experiment configurations."""
    experiment_grid = []
    
    for dataset in DATASETS:
        for method in COUNTERFACTUAL_METHODS:
            for constraint in FAIRNESS_CONSTRAINTS:
                for adj_strength in ADJUSTMENT_STRENGTHS:
                    for amp_factor in AMPLIFICATION_FACTORS:
                        # Create a unique experiment ID
                        exp_id = f"{dataset}_{method}_{constraint}_adj{adj_strength}_amp{amp_factor}"
                        
                        # Create configuration
                        config = BASE_TSCFM_CONFIG.copy()
                        config.update({
                            "dataset": dataset,
                            "counterfactual_method": method,
                            "fairness_constraint": constraint,
                            "adjustment_strength": adj_strength,
                            "amplification_factor": amp_factor,
                            "experiment_id": exp_id
                        })
                        
                        # Add model name for display
                        from utils import make_model_name
                        config["model_name"] = make_model_name(config)
                        
                        experiment_grid.append(config)
    
    return experiment_grid

def get_metrics_list():
    """Return a list of all metrics to calculate and track."""
    performance_metrics = ["BA", "Acc"]
    fairness_metrics = PRIMARY_FAIRNESS_METRICS + SECONDARY_FAIRNESS_METRICS
    return performance_metrics + fairness_metrics

def get_output_dir(experiment_id, base_dir=DEFAULT_OUTPUT_DIR):
    """Get output directory for a specific experiment."""
    output_dir = os.path.join(base_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
