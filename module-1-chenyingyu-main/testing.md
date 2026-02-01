## Testing Your Implementation

### Running Tests

This project uses pytest for testing. Tests are organized by task:

```bash
# Run all tests for a specific task
pytest -m task1_1  # Central difference and numerical derivatives
pytest -m task1_2  # Scalar operations and wrapping
pytest -m task1_3  # Chain rule and computational graphs
pytest -m task1_4  # Backpropagation and gradient computation

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_scalar.py        # Scalar computation tests
pytest tests/test_autodiff.py      # Automatic differentiation tests
pytest tests/test_operators.py     # Basic operators (from Module 0)
pytest tests/test_module.py        # Module system tests (from Module 0)

# Run a specific test function
pytest tests/test_scalar.py::test_central_difference
pytest tests/test_autodiff.py::test_chain_rule
```

### Module 1 Specific Tests

**Task 1.1 - Central Difference:**
- Tests numerical derivative approximation
- Verifies central difference implementation accuracy
- Checks edge cases with different epsilon values

**Task 1.2 - Scalar Operations:**
- Tests scalar wrapping and unwrapping
- Verifies mathematical operations on scalar objects
- Checks that scalar operations maintain gradient tracking

**Task 1.3 - Chain Rule:**
- Tests computational graph construction
- Verifies forward pass through complex expressions
- Checks that intermediate variables are properly tracked

**Task 1.4 - Backpropagation:**
- Tests gradient computation accuracy
- Verifies backpropagation through complex graphs
- Checks gradient accumulation for variables used multiple times

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

### Understanding Test Output

**Property Testing with Hypothesis:**
- Tests use hypothesis to generate random inputs
- If a test fails, Hypothesis will show you the minimal failing example
- This helps you understand edge cases in your autodiff implementation

**Common Test Failures:**
- `AssertionError`: Your function returned an unexpected value or gradient
- `TypeError`: Missing or incorrect type annotations
- `ImportError`: Function not implemented or incorrectly named
- `AttributeError`: Missing methods in scalar or function classes

**Gradient Testing:**
- Many tests compare your computed gradients against numerical approximations
- Small differences (< 1e-5) are usually acceptable due to floating point precision
- Large differences indicate errors in your derivative implementations

### Pre-commit Hooks (Automatic Style Checking)

The project uses pre-commit hooks that run automatically before each commit:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Now style checks run automatically on every commit
git commit -m "your message"  # Will run style checks first
```

### GitHub Classroom Autograder

The autograder runs the same tests and style checks:

1. **Style Check (10 points)**: All pre-commit hooks must pass
2. **Task 1.1 (15 points)**: Central difference implementation
3. **Task 1.2 (15 points)**: Scalar operations and wrapping
4. **Task 1.3 (15 points)**: Chain rule and computational graphs
5. **Task 1.4 (15 points)**: Backpropagation implementation
6. **Task 1.5 (30 points)**: Training and optimization