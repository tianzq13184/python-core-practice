# Python Core Practice

A comprehensive collection of practice problems covering essential Python libraries and frameworks for data science, machine learning, and web development.

## Overview

This repository contains Jupyter notebooks with hands-on practice problems for mastering core Python libraries. Each notebook focuses on specific concepts and includes TODO exercises to help you build practical skills.

## Contents

### 1. [Requests Practice Problems](requests-practice-problems.ipynb)
HTTP requests and API interactions using the `requests` library.

**Topics covered:**
- HTTP basics: GET/POST, query/body/header, status codes, timeout/retry
- Sessions and authentication: Session, cookies, tokens
- Data formats: JSON encoding/decoding, file upload/download
- Reliability: timeout, retry, idempotency, exception handling
- Performance and security: connection reuse, proxy, SSL certificate verification

### 2. [NumPy Practice Problems](numpy-practice-problems.ipynb)
Essential NumPy operations for numerical computing.

**Topics covered:**
- ndarray concepts: shape/dtype/axis, view vs copy
- Broadcasting mechanism
- Vectorization and performance: avoiding Python loops
- Sorting and indexing: boolean indexing, advanced indexing, np.where
- Statistics and linear algebra: sum/mean/std, dot product, norm

### 3. [Pandas Practice Problems](pandas-practice-problems.ipynb)
Data manipulation and analysis with Pandas.

**Topics covered:**
- Data types and missing values: dtype, NA/NaN, missing value handling strategies
- Index system: index/columns, multi-level index (concept)
- Transformation pipeline: select/filter/assign, apply/map vs vectorized operations
- Aggregation: groupby aggregations, rolling windows (concept)
- Merging: merge/join/concat, join keys and duplicate row risks
- Time series: datetime operations, resample (concept)
- Performance and memory: categorical data types, chunk processing, avoiding apply abuse

### 4. [Matplotlib Practice Problems](matplotlib-practice-problems.ipynb)
Data visualization with matplotlib.

**Topics covered:**
- Chart selection: line plots, bar charts, histograms, box plots, scatter plots
- Multiple axes and subplots (concept)
- Annotations: title/label/legend/annotation
- Reading charts: verifying distributions, identifying outliers, showing trends, comparing groups
- Output capabilities: saving figures, resolution, report-ready visualizations

### 5. [Scikit-learn Practice Problems](scikit-learn-practice-problems.ipynb)
Machine learning with scikit-learn.

**Topics covered:**
- Pipeline thinking: preprocess + model + evaluation
- Data splitting: train/valid/test, cross-validation (concept)
- Feature processing: encoding, standardization, missing values, data leakage risks
- Common models: linear models, tree models, ensemble models
- Evaluation metrics: classification/regression metrics, threshold and imbalanced data handling
- Overfitting and hyperparameter tuning: regularization, early stopping (concept), grid/random search
- Interpretability and diagnostics: feature importance, error analysis

### 6. [FastAPI Practice Problems](fastapi-practice-problems.ipynb)
Building REST APIs with FastAPI.

**Topics covered:**
- API design: REST resources, status codes, error response formats
- Pydantic validation: schema definition, type constraints, error messages
- Dependency injection: Depends (concept), authentication, configuration injection
- Asynchronous programming: async/await, I/O bound operations, concurrency model
- Middleware: logging, trace ID, CORS, rate limiting (concept)
- Deployment: Uvicorn/Gunicorn (concept), health checks, monitoring metrics (concept)
- Testing: API testing, contract stability

### 7. [PyTorch Practice Problems](pytorch-practice-problems.ipynb)
Deep learning with PyTorch.

**Topics covered:**
- Tensor basics: device/dtype/shape, broadcasting, in-place operation risks
- Autograd: computational graph, requires_grad, detach (concept)
- Training loop: forward/backward/step/zero_grad
- DataLoader: batching, shuffle, num_workers (concept)
- Common modules: nn.Module, loss functions, optimizers, schedulers (concept)
- Training stability: random seed, gradient explosion/vanishing, gradient clipping (concept)
- Performance and deployment: mixed precision (concept), TorchScript/ONNX (concept)

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required packages:
  ```bash
  pip install requests numpy pandas matplotlib scikit-learn fastapi uvicorn torch
  ```

### Usage

1. Clone this repository:
   ```bash
   git clone git@github.com:tianzq13184/python-core-practice.git
   cd python-core-practice
   ```

2. Open any notebook in Jupyter:
   ```bash
   jupyter notebook
   ```

3. Complete the exercises marked with `# TODO` in each cell.

4. Run the cells to verify your solutions match the expected output.

## Structure

```
python-core-practice/
├── README.md
├── python-algo-practice-problems.ipynb
├── requests-practice-problems.ipynb
├── numpy-practice-problems.ipynb
├── pandas-practice-problems.ipynb
├── matplotlib-practice-problems.ipynb
├── scikit-learn-practice-problems.ipynb
├── fastapi-practice-problems.ipynb
└── pytorch-practice-problems.ipynb
```

## Learning Path

1. **Beginner**: Start with Requests and NumPy to understand HTTP requests and numerical computing basics.
2. **Intermediate**: Move to Pandas and Matplotlib for data analysis and visualization.
3. **Advanced**: Explore Scikit-learn for machine learning, FastAPI for web development, and PyTorch for deep learning.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Author

Created for Python core practice and skill development.

