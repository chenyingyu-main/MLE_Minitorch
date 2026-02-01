## Testing Your Implementation

### Running Tests

This project uses pytest for testing. Tests are organized by task:

```bash
# Run all tests for a specific task
pytest -m task2_1  # Tensor data and indexing
pytest -m task2_2  # Tensor broadcasting
pytest -m task2_3  # Tensor operations
pytest -m task2_4  # Tensor autodifferentiation

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_tensor_data.py    # Tensor data structure tests
pytest tests/test_tensor.py         # Tensor operations and autodiff tests
pytest tests/test_operators.py      # Basic operators (from Module 0)
pytest tests/test_module.py         # Module system tests (from Module 0)
pytest tests/test_scalar.py         # Scalar tests (from Module 1)
pytest tests/test_autodiff.py       # Autodiff tests (from Module 1)

# Run a specific test function
pytest tests/test_tensor_data.py::test_index_to_position
pytest tests/test_tensor.py::test_tensor_sum
```

### Module 2 Specific Tests

**Task 2.1 - Tensor Data:**
- Tests tensor indexing and storage management
- Verifies stride calculations and memory layout
- Checks permutation operations
- Tests `index_to_position` and `to_index` functions

**Task 2.2 - Tensor Broadcasting:**
- Tests broadcasting rules for different tensor shapes
- Verifies `shape_broadcast` and `broadcast_index` functions
- Checks edge cases with dimension alignment
- Tests operations between tensors of different sizes

**Task 2.3 - Tensor Operations:**
- Tests high-level tensor operations (map, zip, reduce)
- Verifies mathematical functions (add, mul, sigmoid, relu, etc.)
- Checks tensor creation and manipulation
- Tests tensor properties and methods

**Task 2.4 - Tensor Autodifferentiation:**
- Tests gradient computation through tensor operations
- Verifies backpropagation with broadcasting
- Checks gradient accumulation and chain rule
- Tests complex computational graphs with tensors

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
- Tests use hypothesis to generate random tensor shapes and values
- If a test fails, Hypothesis will show you the minimal failing example
- This helps you understand edge cases in your tensor implementation

**Common Test Failures:**
- `AssertionError`: Your function returned an unexpected tensor or gradient
- `TypeError`: Missing or incorrect type annotations
- `ImportError`: Function not implemented or incorrectly named
- `AttributeError`: Missing methods in tensor classes
- `IndexingError`: Issues with tensor indexing or broadcasting

**Gradient Testing:**
- Many tests compare your computed gradients against numerical approximations
- Small differences (< 1e-5) are usually acceptable due to floating point precision
- Large differences indicate errors in your derivative implementations

**Broadcasting Errors:**
- Tests will check that tensors with incompatible shapes raise appropriate errors
- Verify that your broadcasting functions handle edge cases correctly

### Task 2.5 - Training

**Training Script:**
```bash
# Run tensor-based training
python project/run_tensor.py
```

**Expected Output:**
- Should train faster than scalar implementation
- Record time per epoch for performance comparison
- Train on all datasets: Simple, Diag, Split, Xor

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
2. **Task 2.1 (15 points)**: Tensor data and indexing implementation
3. **Task 2.2 (15 points)**: Tensor broadcasting implementation
4. **Task 2.3 (15 points)**: Tensor operations implementation
5. **Task 2.4 (15 points)**: Tensor autodifferentiation implementation
6. **Task 2.5 (30 points)**: Training and performance verification

### Debugging Tools

**Interactive Debugging:**
```bash
# Launch tensor visualization app
streamlit run project/app.py -- 2

# Test specific tensor operations
python -c "from minitorch import tensor; t = tensor([1,2,3]); print(t)"
```

**Performance Testing:**
- Compare training times between scalar and tensor implementations
- Verify that tensor operations are significantly faster
- Monitor memory usage with larger tensor operations