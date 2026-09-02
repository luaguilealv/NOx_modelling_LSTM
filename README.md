# NOx Emissions Prediction Using Recurrent Neural Networks (LSTM) in MATLAB

This repository contains a comprehensive MATLAB framework for predicting Nitrogen Oxide (**NOx**) emissions in industrial time-series processes based on operational variables.

The project covers the entire pipeline: from data preprocessing and feature selection to a **multi-criteria optimization methodology** (balancing computational cost in **FLOPs**, predictive accuracy, network architecture, and loss functions), as well as **real-time inference and evaluation** on unseen datasets.

## Key Features

- **Correlation-Based Feature Selection**: Automatic selection of input variables whose absolute correlation coefficient with the target variable ($\text{NOx}$) exceeds a configurable threshold ($\vert{}r\vert{} > 0.4$).
- **Multivariate & Multi-Loss Benchmarking**: Direct comparison among loss functions: **MSE**, **MAE**, and **Huber Loss** (robust against outliers).
- **Architecture Evaluation**:
  - *Regularized Architecture (`Dropout_FC16`)*: Includes Dropout layers (0.2) to mitigate overfitting.
  - *Deep Dense Architecture (`FC32_FC16_ReLU`)*: Cascaded Fully Connected layers with ReLU activations to capture complex non-linearities.
- **FLOP-Constrained Architecture Optimizer**: Grid Search over the hyperparameter space $(T, H)$ constrained by a maximum computational threshold ($\le 5 \times 10^6$ FLOPs).
- **Tolerance-Based Selection ($\epsilon$-Selection)**: Automated identification of the lightest network within an acceptable performance degradation window ($\epsilon = 0.02$).
- **In-Production Real-Time Inference Pipeline**: Dedicated module to apply trained models to new datasets without data leakage.
- **Domain-Specific Industrial Metrics**:
  - Global performance: MSE, RMSE, MAE, $R^2$.
  - Operational risk metrics: 95th Percentile Error ($P_{95}$), Maximum Absolute Error, and **Error at True Peak NOx Value**.

## Mathematical Formulation

### 1. Computational Cost Estimation (FLOPs)
To guarantee deployment feasibility on embedded emission control systems, the computational complexity of the LSTM layer is modeled as:

$$\text{FLOPs} \approx T \times H \times (4d + 4H + 3)$$

Where:
- $T$: Sequence time length (Sliding window size).
- $H$: Number of hidden units in the recurrent LSTM layer.
- $d$: Number of selected input features.

### 2. Tolerance Selection Criterion ($\epsilon$)
Among all candidates satisfying the FLOPs limit ($\le 5 \times 10^6$), the minimum validation loss $\text{ValLoss}_{\min}$ is determined. The selected architecture for each *(Loss, Architecture)* pair satisfies:

$$\text{ValLoss} \le \text{ValLoss}_{\min} + \epsilon \quad \implies \quad \arg\min_{\text{candidates}} (\text{FLOPs})$$

## Evaluated Architectures

| Component | Architecture A (`Dropout_FC16`) | Architecture B (`FC32_FC16_ReLU`) |
| :--- | :--- | :--- |
| **Input Layer** | `sequenceInputLayer(d)` | `sequenceInputLayer(d)` |
| **Recurrent Layer** | `lstmLayer(H, OutputMode="last")` | `lstmLayer(H, OutputMode="last")` |
| **Dense Layer 1** | `dropoutLayer(0.2)` | `fullyConnectedLayer(32)` + `reluLayer` |
| **Dense Layer 2** | `fullyConnectedLayer(16)` | `fullyConnectedLayer(16)` + `reluLayer` |
| **Regularization / Output** | `dropoutLayer(0.2)` $\rightarrow$ `fullyConnectedLayer(1)` | `fullyConnectedLayer(1)` |
| **Focus** | Overfitting prevention on noisy signals | Non-linear feature mapping |

## Repository Structure & Script Directory

| Script Name | Role & Description |
| :--- | :--- |
| `train.m` | **Baseline Training**: Filters features by correlation, builds the sequence dataset, trains a single baseline LSTM model, evaluates test metrics, and exports `model.mat`. |
| `flop_small_search.m` | **Single-Architecture FLOP Search**: Performs a grid search over sequence length ($T$) and hidden units ($H$) for a single architecture to analyze the accuracy vs. FLOPs trade-off. |
| `flop_search.m` | **Exhaustive Multi-Loss & Multi-Arch Search**: Runs a massive GPU-accelerated grid search across multiple loss functions (MSE, MAE, Huber) and network topologies under a strict FLOP limit. Exports summary tables (`.xlsx`). |
| `RT_try.m` | **Real-Time Inferences & Out-of-Sample Testing**: Loads a trained `.mat` model artifact, applies saved preprocessing parameters to new input data, and performs real-time predictions with comprehensive diagnostic plots. |

## Project Workflow

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         1. TRAINING PHASE                              │
 └────────────────────────────────────────────────────────────────────────┘
                    [Base Dataset: var_selec_data.xlsx]
                                   │
                                   ▼
             [Correlation Filtering (|r| > 0.4) -> d vars]
                                   │
                                   ▼
             [Chronological Split: 70% Train / 15% Val / 15% Test]
                                   │
                                   ▼
             [Z-score Standardization (Store µX, σX, µy, σy)]
                                   │
                                   ▼
             [FLOP Optimization & Multi-Loss Benchmarking]
                                   │
                                   ▼
             [Artifact Saving: NOx_model_1.mat]

 ┌────────────────────────────────────────────────────────────────────────┐
 │                    2. REAL-TIME INFERENCE PHASE                        │
 └────────────────────────────────────────────────────────────────────────┘
                  [New Dataset: var_selec_data3.xlsx]
                                   │
                                   ▼
              [Load Artifacts: net, µ, σ, selectedIdx, T]
                                   │
                                   ▼
              [Select identical features from input data]
                                   │
                                   ▼
              [Standardize using TRAINING µX and σX]
                                   │
                                   ▼
              [Build Time-Series Sequences (makeSequences)]
                                   │
                                   ▼
              [Predict with minibatchpredict -> Unstandardize]
                                   │
                                   ▼
              [Generate Metrics & Plots (Real vs. Pred, Residuals)] 
