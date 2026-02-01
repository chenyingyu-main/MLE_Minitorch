## Testing Your Implementation

### Running Tests

This project uses pytest for testing. Tests are organized by task:

```bash
# Run all tests for a specific task
pytest -m task0_1  # Basic operators
pytest -m task0_2  # Property tests
pytest -m task0_3  # Higher-order functions
pytest -m task0_4  # Module system

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_operators.py
pytest tests/test_module.py

# Run a specific test function
pytest tests/test_operators.py::test_same_as_python
```
Note that the `task0_1` test will not fully pass until you complete `task0_3`.

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
- This helps you understand edge cases in your implementation

**Common Test Failures:**
- `AssertionError`: Your function returned an unexpected value
- `TypeError`: Missing or incorrect type annotations
- `ImportError`: Function not implemented or incorrectly named

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
2. **Task 0.1 (10 points)**: Basic mathematical operators
3. **Task 0.2 (10 points)**: Property-based testing
4. **Task 0.3 (10 points)**: Higher-order functions
5. **Task 0.4 (10 points)**: Module system
