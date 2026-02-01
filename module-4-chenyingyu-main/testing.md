## Testing Your Implementation

### Running Tests

This project uses pytest for testing. Tests are organized by task:

```bash
# Module 4 Tasks - Run locally
pytest -m task4_1  # 1D convolution
pytest -m task4_2  # 2D convolution
pytest -m task4_3  # Pooling operations
pytest -m task4_4  # Advanced NN functions

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_conv.py     # Convolution tests
pytest tests/test_nn.py       # Neural network tests

# Run a specific test function
pytest tests/test_conv.py::test_conv1d_simple
pytest tests/test_nn.py::test_softmax
```

### MNIST Dataset Testing

**Module 4 requires MNIST dataset for CNN training:**

Before running CNN training tests, ensure MNIST is properly installed:
```bash
# Verify MNIST dataset
python -c "import mnist; print('MNIST available')"

# Test MNIST loading in MiniTorch
python project/run_mnist_multiclass.py
```

### Style and Code Quality Checks

This project enforces code style and quality using several tools:

```bash
# Run all pre-commit hooks (recommended)
pre-commit run --all-files

# Individual style checks:
ruff check .                 # Linting (style, imports, docstrings)
ruff format .               # Code formatting
pyright .                   # Type checking
```

### Pre-commit Hooks (Automatic Style Checking)

The project uses pre-commit hooks that run automatically before each commit:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Now style checks run automatically on every commit
git commit -m "your message"  # Will run style checks first
```