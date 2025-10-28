# 🧪 PPL Benchmark
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**PPL Benchmark** is an open-source framework for benchmarking probabilistic programming languages (PPLs), currently supporting **[Pyro](https://pyro.ai)** and **[NumPyro](https://num.pyro.ai)**.  
It provides a unified interface to evaluate performance, accuracy, and scalability across inference methods and Bayesian models.

---

## 🚀 Motivation

Probabilistic programming frameworks like Pyro and NumPyro make it easy to define and infer complex Bayesian models. However, comparing their **performance** and **scalability** under consistent conditions can be challenging. PPL Benchmark provides a lightweight framework that performs specified benchmarks across PPL-Frameworks and inference methods. It is designed to enable rapid integration of new models and automatically perform the benchmarks on them on all supported frameworks. 

### Probabilistic Programming Languages
- **Pyro**: A deep universal probabilistic programming language built on PyTorch.
- **NumPyro**: A lightweight Pyro-like library built on JAX.

### Inference Algorithms
- **Stochastic Variational Inference (SVI)**: A fast, optimization-based inference method.
- **Markov Chain Monte Carlo (MCMC)**: A more accurate but computationally intensive sampling-based method.

### PPL Benchmark was created to
- Provide a **common benchmarking framework** for multiple PPLs.
- Offer **Bayesian models** for fair comparisons.
- Measure **runtime, inference accuracy, and number of model evaluations** across backends.
- Simplify **experiment management** with configurable YAML files.

---

## 🛠️ Key Features

- **Performance Metrics**: Measures execution time, forward model calls, and backward/gradient computations
- **Multiple Models**: Up to now, includes implementations of Bayesian Linear Regression and the Eight Schools model
- **Cross-library Comparison**: Direct comparison between Pyro and NumPyro implementations
- **Extensible Design**: Easy to add new models or benchmark metrics
- **Reproducible Results**: Configurable parameters for consistent benchmarking


---

## 📂 Project Structure
The project is structured into modular components

```
PPL_Benchmark/
│
├── ppl_benchmark/
│   ├── core/                  # Benchmark and inference logic
│   ├── models/                # Implemented Bayesian models
│   ├── utils/                 # Helper functions and logging
│   ├── config/                # Experiment configuration files
│   ├── visualisation/         # Plotting and results visualization
│   └── main.py                # CLI entry point
│
├── tests/                     # Unit tests
└── README.md                  # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/mclprobability/PPLBenchmark.git
cd PPL_Benchmark
```

### 2. Set up an anaconda environment (recommended)
```bash
conda init                # activate anaconda in the shell
conda create -n ppl_benchmark python=3.11
conda activate ppl_benchmark     
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


---

## 🧪 Usage

### Run a benchmark
After installation, you can run the benchmark with default settings on a simple BayesianLinearRegression model:

```bash
python -m ppl_benchmark
```

The default YAML configuration [ppl_benchmark.yml](ppl_benchmark/config/base/ppl_benchmark.yml) defines:
- Sampling parameters
- Logging and output directories

### Or use it programmatically in Python:

```python
import torch
import pyro
from ppl_benchmark.core.benchmark import benchmark_model
from ppl_benchmark.config import Config, Framework, InferenceRoutine
from ppl_benchmark.models.bayesian_regression import BayesianRegressionPyro

# Create a configuration
cfg = Config()
cfg.svi_iterations = 100

# Create a model & guide for SVI in pyro
model = BayesianRegressionPyro()
guide = pyro.infer.autoguide.AutoNormal(model)

# create dummy test data
N=10
x_demo = torch.randn((N,3))
y_demo = torch.randn((N))

# Run the benchmark
result = benchmark_model(
    Framework.PYRO, 
    InferenceRoutine.SVI, 
    cfg.__dict__, 
    model, 
    x_demo, 
    y_demo,
    guide=guide
)

# Print results
print(f"Execution time: {result.execution_time:.3f} seconds")
print(f"Forward calls: {result.forward_calls}")
print(f"Backward calls: {result.backward_calls}")
```
A more complete version of this example is provided in the project [main.py](ppl_benchmark/main.py)

---

## 🧷 Testing
Run the unit tests to ensure everything is functioning correctly:
```bash
pytest tests/
```

---

## 🧑‍💻 Contributing

Contributions are welcome!  
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

If you’re adding a new model or backend, include a test under `tests/` and update the documentation accordingly.

---

## 🪪 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


---

### 🧭 Future Work
- Extend the number of benchmark models and datasets
  - include the models of [PosteriorDB](https://github.com/stan-dev/posteriordb) that have a reference posterior
- safe the results in the respective results folders of the experiments
