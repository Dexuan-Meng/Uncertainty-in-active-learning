# Uncertainty Estimation in Active Learning

This repository contains the implementation for the course project **"A Comparison of Methods for Uncertainty Estimation in Active Learning"**. The goal is to evaluate and compare multiple uncertainty estimation methods under the active learning framework using the BALD (Bayesian Active Learning by Disagreement) query strategy.

## Overview

In many machine learning scenarios, obtaining labeled data is expensive and time-consuming. Active Learning (AL) seeks to reduce labeling effort by selecting the most informative samples. This project investigates and compares different methods of uncertainty estimation when used with BALD in an active learning pipeline.

## Methods Compared

We compare the following uncertainty estimation approaches:

- **Gaussian Process (GP)**
- **Ensemble-based Methods (Random Forest, Ensemble CNN)**
- **Monte Carlo Dropout (MC-Dropout)**
- **Bayesian Neural Networks (Bayes by Backprop / Bayesian CNN)**

All methods are tested on the MNIST dataset in an image classification setting.

## Acquisition Strategy

- **BALD**: Selects points that maximize mutual information between predictions and model posterior.
- A baseline using **uniform sampling** is also used for comparison.

## Evaluation Metric

**Relative Exceeding (RE)**:  
RE = (sum of (Bᵢ - Uᵢ)) / (Bₙ - B₁)  

Where:  
- Bᵢ: Accuracy after the i-th query using BALD  
- Uᵢ: Accuracy after the i-th query using Uniform sampling  
- B₁ and Bₙ: Accuracy after the first and last query steps using BALD

## Experiments

- Two initial dataset sizes tested: **50** and **1000** samples
- **100 query steps**, each querying **1 instance**
- Experiments performed on MNIST, results visualized and tabulated

## Key Findings

- **MC-Dropout** consistently performed best across different dataset sizes
- **Bayesian CNN** performed well with a large initial dataset but failed when data was limited
- **Gaussian Process CNN** and **Ensemble CNN** showed good performance with small initial datasets but plateaued with more data
- BALD tends to select redundant samples; using a single-instance query helps reduce this issue

## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- numpy
- matplotlib

## File Structure

```
.
├── data/                   # MNIST dataset or loaders
├── models/                 # CNN, Bayesian CNN, GP, Ensemble etc.
├── utils/                  # Helper functions for evaluation and plotting
├── main.py                 # Main script to run active learning experiment
├── config.yaml             # Configurations for model training and experiments
└── README.md
```

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run experiments:
   ```bash
   python main.py --config config.yaml
   ```

3. Evaluate results and generate plots.

## Citation

If you use this project or parts of it in your work, please cite:

> Dexuan Meng, Chenyu Meng. *A Comparison of Methods for Uncertainty Estimation in Active Learning*, TUM, 2023.

## License

MIT License

---

For questions or contributions, feel free to contact:  
📧 ge28daw@mytum.de  
