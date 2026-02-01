# MiniTorch Module 2 Installation

MiniTorch requires Python 3.8 or higher. To check your version of Python, run:

```bash
>>> python --version
```

We recommend creating a global MiniTorch workspace directory that you will use
for all modules:

```bash
>>> mkdir workspace; cd workspace
```

## Environment Setup

We highly recommend setting up a *virtual environment*. The virtual environment lets you install packages that are only used for your assignments and do not impact the rest of the system.

**Option 1: Anaconda (Recommended)**
```bash
>>> conda create --name minitorch python    # Run only once
>>> conda activate minitorch
>>> conda install llvmlite                  # For optimization
```

**Option 2: Venv**
```bash
>>> python -m venv venv          # Run only once
>>> source venv/bin/activate
```

The first line should be run only once, whereas the second needs to be run whenever you open a new terminal to get started for the class. You can tell if it works by checking if your terminal starts with `(minitorch)` or `(venv)`.

## Getting the Code

Each assignment is distributed through a Git repo. Once you accept the assignment from GitHub Classroom, a personal repository under Cornell-Tech-ML will be created for you. You can then clone this repository to start working on your assignment.

```bash
>>> git clone {{ASSIGNMENT}}
>>> cd {{ASSIGNMENT}}
```

## Syncing Previous Module Files

Module 2 requires files from Module 0 and Module 1. Sync them using:

```bash
>>> python sync_previous_module.py <path-to-module-1> <path-to-current-module>
```

Example:
```bash
>>> python sync_previous_module.py ../Module-1 .
```

Replace `<path-to-module-1>` with the path to your Module 1 directory and `<path-to-current-module>` with `.` for the current directory.

This will copy the following required files:
- `minitorch/operators.py`
- `minitorch/module.py`
- `minitorch/autodiff.py`
- `minitorch/scalar.py`
- `tests/test_module.py`
- `tests/test_operators.py`
- `tests/test_autodiff.py`
- `tests/test_scalar.py`
- `project/run_manual.py`
- `project/run_scalar.py`

## Installation

Install all packages in your virtual environment:

```bash
>>> python -m pip install -e ".[dev,extra]"
```

## Verification

Make sure everything is installed by running:

```bash
>>> python -c "import minitorch; print('Success!')"
```

Verify that the tensor functionality is available:

```bash
>>> python -c "from minitorch import tensor; print('Module 2 ready!')"
```

You're ready to start Module 2!