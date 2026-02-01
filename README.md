# MiniTorch

A from-scratch reimplementation of the core PyTorch API, built as part of the [MiniTorch](https://minitorch.github.io/) project from Cornell Tech's Machine Learning Engineering course.

The goal is to understand how a modern deep learning framework works under the hood — from basic math operators all the way up to training CNNs — by implementing every piece yourself in pure Python (and CUDA).

---

## Project Structure

```
minitorch/
├── module-1/          # Autodifferentiation (Scalar)
├── module-2/          # Tensors
├── module-3/          # Efficiency & GPU
├── module-4/          # Neural Networks & CNNs
└── Notebook_GPU_Test/ # Google Colab notebooks for GPU acceleration testing
```

---

## Modules

### Module 1 — Autodifferentiation

The foundation of any deep learning framework: automatic differentiation. This module implements scalar-level autodiff from scratch.

- Implement basic scalar math functions (`ScalarFunction`) with both forward and backward passes
- Derive the chain rule manually and use it to build a computational graph
- Implement backpropagation on the scalar graph to compute gradients automatically

Core files: `scalar.py`, `scalar_functions.py`, `autodiff.py`

---

### Module 2 — Tensors

Scale everything up from scalars to tensors. This is where the real data structures come in.

- Design `TensorData`: the underlying storage layout (strides, shape, storage) that makes tensor operations efficient
- Implement broadcasting so that operations work across tensors of different shapes (following NumPy-style rules)
- Port all the scalar operations to work on full tensors, and wire up autograd so gradients flow through tensor ops

Core files: `tensor_data.py`, `tensor_ops.py`, `tensor_functions.py`, `tensor.py`

---

### Module 3 — Efficiency

Raw Python loops are way too slow for real ML workloads. This module is about making things fast.

- Optimize tensor operations using NumPy-backed vectorized ops (`fast_ops.py`) to replace naive Python loops
- Write CUDA kernels in raw CUDA C to run tensor operations on the GPU (`cuda_ops.py`)
- Implement efficient matrix multiplication — the single most important operation in deep learning — on both CPU and GPU

Core files: `fast_ops.py`, `cuda_ops.py`

---

### Module 4 — Neural Networks

Put it all together and build actual neural network architectures.

- Implement 1D and 2D convolution (`Conv1d`, `Conv2d`) and pooling layers (`maxpool2d`) from scratch
- Add modern training utilities: `softmax`, `logsoftmax`, `dropout`
- Train a **LeNet-style CNN on MNIST** for handwritten digit classification
- Train a **CNN-based sentiment classifier on SST-2** using a Kim (2014)-style architecture with parallel conv branches

Core files: `nn.py`, `cuda_conv.py`, `project/run_mnist_multiclass.py`, `project/run_sentiment.py`

---

### Notebook\_GPU\_Test

Google Colab notebooks used to test and validate GPU-accelerated operations. Useful for verifying that CUDA kernels produce correct results and comparing GPU vs CPU performance on tensor ops and convolutions.

---

## Setup

Requires **Python 3.11+**. It's recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU support (Module 3 & 4), a CUDA-capable GPU and the CUDA toolkit are needed. The Colab notebooks in `Notebook_GPU_Test/` can be used as an alternative.

---

## References

- [MiniTorch Official Site](https://minitorch.github.io/)
- Course: *Machine Learning Engineering*, Cornell Tech — taught by [Ramtin Keramati](https://rkeramati.github.io/)
