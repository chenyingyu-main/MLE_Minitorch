"""Collection of the core mathematical operators used throughout the code base.

This module implements fundamental mathematical operations that serve as building blocks
for neural network computations in MiniTorch.

NOTE: The `task0_1` tests will not fully pass until you complete `task0_3`.
Some tests depend on higher-order functions implemented in the later task.
"""

# =============================================================================
# Task 0.1: Mathematical Operators
# =============================================================================
import math
from typing import Callable, Iterable


# """
# Implementation of elementary mathematical functions.

# FUNCTIONS TO IMPLEMENT:
#     Basic Operations:
#     - mul(x, y)     → Multiply two numbers
#     - id(x)         → Return input unchanged (identity function)
#     - add(x, y)     → Add two numbers
#     - neg(x)        → Negate a number

#     Comparison Operations:
#     - lt(x, y)      → Check if x < y
#     - eq(x, y)      → Check if x == y
#     - max(x, y)     → Return the larger of two numbers
#     - is_close(x, y) → Check if two numbers are approximately equal

#     Activation Functions:
#     - sigmoid(x)    → Apply sigmoid activation: 1/(1 + e^(-x))
#     - relu(x)       → Apply ReLU activation: max(0, x)

#     Mathematical Functions:
#     - log(x)        → Natural logarithm
#     - exp(x)        → Exponential function
#     - inv(x)        → Reciprocal (1/x)

#     Derivative Functions (for backpropagation):
#     - log_back(x, d)  → Derivative of log: d/x
#     - inv_back(x, d)  → Derivative of reciprocal: -d/(x²)
#     - relu_back(x, d) → Derivative of ReLU: d if x>0, else 0

# IMPORTANT IMPLEMENTATION NOTES:

# Numerically Stable Sigmoid:
#    To avoid numerical overflow, use different formulations based on input sign:

#    For x ≥ 0:  sigmoid(x) = 1/(1 + exp(-x))
#    For x < 0:  sigmoid(x) = exp(x)/(1 + exp(x))

#    Why? This prevents computing exp(large_positive_number) which causes overflow.

# is_close Function:
#    Use tolerance: |x - y| < 1e-5
#    This handles floating-point precision issues in comparisons.

# Derivative Functions (Backpropagation):
#    These compute: derivative_of_function(x) × upstream_gradient

#    - log_back(x, d):  d/dx[log(x)] = 1/x  →  return d/x
#    - inv_back(x, d):  d/dx[1/x] = -1/x**2   →  return -d/(x**2)
#    - relu_back(x, d): d/dx[relu(x)] = 1 if x>0 else 0  →  return d if x>0 else 0
# """

# TODO: Implement all functions listed above for Task 0.1


#  Basic Operations:
def mul(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y


def id(x: float) -> float:
    """Return input unchanged (identity function)"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


def neg(x: float) -> float:
    """Negate a number"""
    return -x


# Comparison Operations:
def lt(x: float, y: float) -> float:
    """Check if x < y"""
    return 1 if x < y else 0


def eq(x: float, y: float) -> float:
    """Check if x == y"""
    return 1 if x == y else 0


def max(x: float, y: float) -> float:
    """Return the larger of two numbers"""
    return x if x >= y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are approximately equal"""
    tolerance = 1e-5
    return 1 if abs(x - y) < tolerance else 0


# Activation Functions:
def sigmoid(x: float) -> float:
    """To avoid numerical overflow, use different formulations based on input sign:
    For x ≥ 0:  sigmoid(x) = 1/(1 + exp(-x))
    For x < 0:  sigmoid(x) = exp(x)/(1 + exp(x))
    """
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Derivative of ReLU: d if x>0, else 0"""
    return x if x > 0.0 else 0.0


# Mathematical Functions:
def log(x: float) -> float:
    """Natural logarithm"""
    # if x is a negative number or 0, math.log would raise an error
    # use EPS to avoid -inf
    # EPS = 1e-6

    if x == 0:
        return float("-inf")
    elif x < 0:
        return float("nan")
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Reciprocal (1/x)"""
    if x == 0:
        return float("inf")
    return 1 / x


# Derivative Functions (for backpropagation):
def log_back(x: float, d: float) -> float:
    """Derivative of log: d/x"""
    if x == 0 and d >= 0:
        return float("inf")
    elif x == 0 and d < 0:
        return float("-inf")
    elif x < 0:
        return float("nan")

    return d / x


def inv_back(x: float, d: float) -> float:
    """Derivative of reciprocal: -d/(x²)"""
    if x == 0 and d > 0:
        return float("-inf")
    elif x == 0 and d < 0:
        return float("inf")
    elif x == 0 and d == 0:
        return float("nan")

    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Derivative of ReLU: d if x>0, else 0"""
    return d if x > 0 else 0


# =============================================================================
# Task 0.3: Higher-Order Functions
# =============================================================================

# """
# Implementation of functional programming concepts using higher-order functions.

# These functions work with other functions as arguments, enabling powerful
# abstractions for list operations.

# CORE HIGHER-ORDER FUNCTIONS TO IMPLEMENT:

#     map(fn, iterable):
#         Apply function `fn` to each element of `iterable`
#         Example: map(lambda x: x*2, [1,2,3]) → [2,4,6]

#     zipWith(fn, list1, list2):
#         Combine corresponding elements from two lists using function `fn`
#         Example: zipWith(add, [1,2,3], [4,5,6]) → [5,7,9]

#     reduce(fn, iterable, initial_value):
#         Reduce iterable to single value by repeatedly applying `fn`
#         Example: reduce(add, [1,2,3,4], 0) → 10

# FUNCTIONS TO BUILD USING THE ABOVE:

#     negList(lst):
#         Negate all elements in a list
#         Implementation hint: Use map with the neg function

#     addLists(lst1, lst2):
#         Add corresponding elements from two lists
#         Implementation hint: Use zipWith with the add function

#     sum(lst):
#         Sum all elements in a list
#         Implementation hint: Use reduce with add function and initial value 0

#     prod(lst):
#         Calculate product of all elements in a list
#         Implementation hint: Use reduce with mul function and initial value 1
# """

# TODO: Implement all functions listed above for Task 0.3


# CORE HIGHER-ORDER FUNCTIONS:
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Reference: https://en.wikipedia.org/wiki/Map_(higher-order_function)
    Apply function `fn` to each element of `iterable`
    Example: map(lambda x: x*2, [1,2,3]) → [2,4,6]
    """

    def apply_fn_to_list(ls: Iterable[float]) -> Iterable[float]:
        return [fn(element) for element in ls]

    return apply_fn_to_list


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combine corresponding elements from two lists using function `fn`
    Example: zipWith(add, [1,2,3], [4,5,6]) → [5,7,9]

    Use the zip function in python: https://www.geeksforgeeks.org/python/zip-in-python/
    """

    def combine_with_fn(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(a, b) for a, b in zip(ls1, ls2)]

    return combine_with_fn


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce iterable to single value by repeatedly applying `fn`
    Example: reduce(add, [1,2,3,4], 0) → 10

    Explanation: start = 0
    0 + [1,2,3,4] = 10
    """

    def reduce_with_start(ls: Iterable[float]) -> float:
        base = start
        for num in ls:
            base = fn(base, num)
        return base

    return reduce_with_start


# FUNCTIONS TO BUILD USING CORE HIGHER-ORDER FUNCTIONS:


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list
    Implementation hint: Use map with the neg function
    """
    neg_list: Callable[[Iterable[float]], Iterable[float]] = map(neg)
    return neg_list(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists
    Implementation hint: Use zipWith with the add function
    """
    add_lists: Callable[[Iterable[float], Iterable[float]], Iterable[float]] = zipWith(
        add
    )
    return add_lists(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list
    Implementation hint: Use reduce with add function and initial value 0
    """
    sum_fun: Callable[[Iterable[float]], float] = reduce(add, 0)
    return sum_fun(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate product of all elements in a list
    Implementation hint: Use reduce with mul function and initial value 1
    """
    prod_fun: Callable[[Iterable[float]], float] = reduce(mul, 1)
    return prod_fun(ls)
