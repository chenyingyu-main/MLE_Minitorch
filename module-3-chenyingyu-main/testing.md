## Testing Your Implementation

### Running Tests

This project uses pytest for testing. Tests are organized by task:

```bash
# CPU Tasks (3.1 & 3.2) - Run locally
pytest -m task3_1  # CPU parallel operations
pytest -m task3_2  # CPU matrix multiplication

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_tensor_general.py  # All optimized tensor tests

# Run a specific test function
pytest tests/test_tensor_general.py::test_one_args -k "fast"
pytest tests/test_tensor_general.py::test_matrix_multiply
```

### GPU Testing Strategy

**CI Limitations:**
- GitHub Actions CI only runs tasks 3.1 and 3.2 (CPU only)
- Tasks 3.3 and 3.4 require local GPU or Google Colab

**GPU Tasks (3.3 & 3.4) - Google Colab (Recommended):**

Follow instructions on the [Google Colab link](https://colab.research.google.com/drive/1gyUFUrCXdlIBz9DYItH9YN3gQ2DvUMsI?usp=sharing) and run tests like this:
```bash
!cd $DIR; python3.11 -m pytest -m task3_3 -v
!cd $DIR; python3.11 -m pytest -m task3_4 -v
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

### Parallel Diagnostics (Tasks 3.1 & 3.2)

**Running Parallel Check:**
```bash
# Verify your parallel implementations
python project/parallel_check.py
```

### Pre-commit Hooks (Automatic Style Checking)

The project uses pre-commit hooks that run automatically before each commit:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Now style checks run automatically on every commit
git commit -m "your message"  # Will run style checks first
```