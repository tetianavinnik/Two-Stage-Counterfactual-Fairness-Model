# Two-Stage Counterfactual Fairness Model (TSCFM)

This repository contains the implementation of a Two-Stage Counterfactual Fairness Model (TSCFM) designed to reduce gender bias in credit scoring.

## Overview

The TSCFM approach employs counterfactual reasoning to improve fairness across protected groups while preserving accuracy. It supports multiple counterfactual generation methods and fairness optimization targets.

## Repository Structure

### Core Implementation

`tscfm.py`: Main implementation of the Two-Stage Counterfactual Fairness Model

`base_models.py`: Base classifiers used in the first modeling stage

`causal_graph.py`: Tools for discovering and encoding causal graphs

`counterfactual_generator.py`: Methods for generating counterfactual examples

`fairness_metrics.py`: Implementations of fairness evaluation metrics

`visualization.py`: Utility functions for visualizing results

### Dataset Loaders

`load_german.py`: Loader for the German Credit dataset

`load_card_credit.py`: Loader for the Home Credit dataset

`load_pakdd.py`: Loader for the PAKDD dataset

`data_processor.py`: Preprocessing and feature transformation pipeline

### Experiment Tools

`run_experiments.py`: Main script for running single experiments

`run_tscfm_experiments.py`: Script for running multiple experiment configurations

`config.py`: Configuration settings for models and experiments

`utils.py`: Helper utilities for experiment execution and management

`compile_results.py`: Tools for compiling and analyzing experiment results

### Analysis and Visualization

`performance.py`: Tools for plotting performance metrics

`fairness.py`: Tools for visualizing fairness metrics

`baseline_results.py`: Scripts for comparing TSCFM against baseline models

`baseline_comparison.py`: Generates comparative visualizations for baselines

`constraint_comparison.py`: Visualization of optimization under different fairness constraints

`heatmap.py`: Generates parameter sensitivity heatmaps

`model_comparison.py`: Compares performance of top-performing models

## Experiment Setup

The experiments evaluate TSCFM across multiple dimensions:

Datasets: German Credit, Card Credit, PAKDD

Counterfactual Methods: Structural Equation, Matching, Generative (Group-Difference Shift)

Fairness Constraints: Demographic Parity (DP), Equal Opportunity (EO), Equalized Odds (EOd)

Parameters: Adjustment Strength (0.1–1.0), Amplification Factor (0.5–3.0)

## Running Experiments

Set up paths to datasets in `config.py`

To run a single experiment: ``` python run_tscfm_experiments.py --mode all --dataset german --method structural_equation ```

To run a full grid search across all settings: ``` python run_tscfm_experiments.py --mode all --dataset german ```

## Generating Visualizations

Set path to results file (generated during running experiments in ./compiled/visualization/performance/)

Generate baseline comparisons: ``` python baseline_comparison.py ```

Compare fairness constraints: ``` python constraint_comparison.py ```

Create parameter sensitivity heatmaps: ``` python heatmap.py ```

Compare top-performing models: ``` python model_comparison.py ```


## Results Summary

Experiments demonstrate that TSCFM generally improves fairness across multiple definitions with minimal accuracy sacrifice compared to baseline models.

Structural Equation methods are most effective for Equal Opportunity and Equalized Odds optimization.

Generative (Group-Difference Shift) methods are highly effective for Demographic Parity optimization.

Optimal parameter ranges: Adjustment Strength (0.7–1.0), Amplification Factor (varies by target metric).
