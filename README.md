# IVQRNN: Instrumental Variable Quantile Regression using Sieve Neural Networks

## Overview

IVQRNN is a PyTorch implementation of Instrumental Variable Quantile Regression (IVQR) using sieve neural networks. It is designed to recover heterogeneous treatment effects in large-scale datasets under a partially linear structural model.

## Key Features

* **Sieve Neural Networks**: Implements the shallow network architecture for semiparametric estimation based on Chen and White (1999).
* **Scalable Optimization**: Provides a Stochastic Gradient Descent (SGD) alternative to the grid search.
* **Hybrid Activations**: Uses specialized activation functions designed for quantile-based structural estimation.
* **Automated Scaling**: Hidden layer dimensions scale with sample size () to ensure statistical consistency.
* **Heterogeneous Effects**: Facilitates estimation across the entire distribution of quantiles ().

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* Scikit-Learn
* Statsmodels
* SciPy

## Repository Structure

* **`IVQRNN_grid.py`**: Implementation using the canonical grid-search optimization method.
* **`IVQRNN_sgd.py`**: A high-performance version using SGD with momentum for large-scale data ().
* **`IVQRNN_lin.py`**: Baseline linear IVQR implementation for comparative analysis.

## Usage

### 1. Grid Search 

Run the grid-search implementation for precise estimation on smaller datasets:

```bash
python IVQRNN_grid.py

```

### 2. SGD Optimization 

Run the SGD implementation for high-dimensional or large-scale data applications:

```bash
python IVQRNN_sgd.py

```

## Theoretical Foundation

This implementation leverages the method of sieves to approximate unknown nuisance functions. It maintains consistency and asymptotic normality for the treatment effect parameter. 

## Citation

Abdurahman, S. (2025). Instrumental variable quantile regression using artificial neural networks. *Working Paper.*
