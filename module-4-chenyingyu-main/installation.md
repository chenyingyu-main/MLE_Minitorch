# MiniTorch Module 4 Installation

MiniTorch requires Python 3.11. To check your version of Python, run:

```bash
>>> python --version
```

If you don't have Python 3.11, install it before proceeding:
- **Mac**: `brew install python@3.11`
- **Ubuntu/Debian**: `sudo apt install python3.11`
- **Windows**: Download from python.org

We recommend creating a global MiniTorch workspace directory that you will use
for all modules:

```bash
>>> mkdir workspace; cd workspace
```

## Environment Setup

We highly recommend setting up a *virtual environment*. The virtual environment lets you install packages that are only used for your assignments and do not impact the rest of the system.

**Option 1: Anaconda (Recommended)**
```bash
>>> conda create --name minitorch python=3.11    # Run only once
>>> conda activate minitorch
>>> conda install llvmlite                       # For optimization
```

**Option 2: Venv**
```bash
>>> python3.11 -m venv venv      # Run only once (requires Python 3.11)
>>> source venv/bin/activate
```

The first line should be run only once, whereas the second needs to be run whenever you open a new terminal to get started for the class. You can tell if it works by checking if your terminal starts with `(minitorch)` or `(venv)`.

## Getting the Code

Each assignment is distributed through a Git repo. Once you accept the assignment from GitHub Classroom, a personal repository under Cornell-Tech-ML will be created for you. You can then clone this repository to start working on your assignment.

```bash
>>> git clone {{ASSIGNMENT}}
>>> cd {{ASSIGNMENT}}
```

## Installation

Install all packages in your virtual environment:

```bash
>>> python -m pip install -e ".[dev,extra]"
```

## Syncing Previous Module Files

Module 4 requires files from Module 0, Module 1, Module 2, and Module 3. Sync them using:

```bash
>>> python sync_previous_module.py <path-to-module-3> <path-to-current-module>
```

Example:
```bash
>>> python sync_previous_module.py ../Module-3 .
```

Replace `<path-to-module-3>` with the path to your Module 3 directory and `<path-to-current-module>` with `.` for the current directory.

This will copy the following required files:
- `minitorch/tensor_data.py`
- `minitorch/tensor_functions.py`
- `minitorch/tensor_ops.py`
- `minitorch/operators.py`
- `minitorch/scalar.py`
- `minitorch/scalar_functions.py`
- `minitorch/module.py`
- `minitorch/autodiff.py`
- `minitorch/tensor.py`
- `minitorch/datasets.py`
- `minitorch/testing.py`
- `minitorch/optim.py`
- `minitorch/fast_ops.py`
- `minitorch/cuda_ops.py`
- `project/run_manual.py`
- `project/run_scalar.py`
- `project/run_tensor.py`
- `project/parallel_check.py`
- `tests/test_tensor_general.py`

## MNIST Dataset Setup

Module 4 requires the MNIST handwritten digit dataset for CNN training. Install and download the dataset:

```bash
>>> pip install python-mnist
>>> mnist_get_data.sh
```

On Mac, this may require installing the `wget` command:
```bash
>>> brew install wget
```

## CUDA Support (Optional - for Task 4.4b Extra Credit)

Task 4.4b requires CUDA and should be completed on **Google Colab** with GPU runtime:

1. **Upload your Module-4 files to Google Colab**
2. **Enable GPU runtime**: Runtime → Change runtime type → Hardware accelerator: GPU
3. **Install in Colab**: Follow the [Module-3 Colab guide](https://colab.research.google.com/drive/1gyUFUrCXdlIBz9DYItH9YN3gQ2DvUMsI?usp=sharing) for CUDA setup, then install Module-4 with CUDA support: