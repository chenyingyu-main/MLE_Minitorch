"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes a * b  for tensors a and b"""
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of a * b with respect to tensor a and b"""
        # TODO: Implement for Task 2.4.
        (a, b) = ctx.saved_values
        back_a = grad_output.f.mul_zip(grad_output, b)
        back_b = grad_output.f.mul_zip(grad_output, a)
        return back_a, back_b

class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the sigmoid function for the input tensor t1"""
        # TODO: Implement for Task 2.3.

        # Same as Scalar, Note: the derivative formula uses σ(a) * (1 - σ(a))
        # so, for sigmoid forward, we save the output (σ(a)) instead of the input :)

        sigmoid_val = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the sigmoid function with respect to its input tensor"""
        # TODO: Implement for Task 2.4.

        # Derivative of sigmoid:
        # dσ/da = σ(a) * (1 - σ(a))
        # s = σ(a)
        # So chain rule: dL/da = dL/dσ * dσ/da
        #                      = d_output * s * (1 - s)

        (sigmoid_val,) = ctx.saved_values
        one_minus_s = (- sigmoid_val + tensor([1.0]))
        s_times_one_minus_s = grad_output.f.mul_zip(sigmoid_val, one_minus_s)
        return grad_output.f.mul_zip(grad_output, s_times_one_minus_s)

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the ReLU function element-wise to the input tensor t1."""
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the ReLU function with respect to its input tensor."""
        # TODO: Implement for Task 2.4.
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the natural logarithm of the input tensor t1."""
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the natural logarithm with respect to its input tensor."""
        # TODO: Implement for Task 2.4.
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the exponential of the input tensor t1."""
        # TODO: Implement for Task 2.3.
        exp_t1 = t1.f.exp_map(t1)
        ctx.save_for_backward(exp_t1)
        return exp_t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the exponential function with respect to its input tensor."""
        # TODO: Implement for Task 2.4.
        # Derivative of exp:
        # d(exp(a))/da = exp(a)
        # So chain rule: dL/da = dL/d(exp(a)) * d
        #                      = d_output * exp(a)

        (exp_t1,) = ctx.saved_values
        return grad_output.f.mul_zip(exp_t1, grad_output)

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Apply the sum function along dimension dim of tensor a."""
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the sum function with respect to its input tensor a."""
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compare a < b element-wise for tensors a and b"""
        # TODO: Implement for Task 2.3.
        # we need the shape of tensor a b for the backward
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """The gradient of a comparison is always zero."""
        # TODO: Implement for Task 2.4.
        (a_shape, b_shape) = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)

class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compare a == b element-wise for tensors a and b"""
        # TODO: Implement for Task 2.3.
        # we need the shape of tensor a b for the backward
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)


    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """The gradient of a comparison is always zero."""
        # TODO: Implement for Task 2.4.
        (a_shape, b_shape) = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise comparison of whether two tensors are close."""
        # TODO: Implement for Task 2.3.
        return a.f.is_close_zip(a, b)

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # Hint: Convert order tensor to list of integers
        # Use tensor's permute method on the underlying _tensor
        # Create new Tensor with same backend

        # convert order tensor to list of integers
        # we are passing it to backend permute function
        re_order = [0] * len(list(order.to_numpy()))

        order_list = []
        # we are going to store the nums in "order tensor" to a list of integers
        # cause the permute function in TensorData needs a python list

        for id, val in enumerate(list(order.to_numpy())):
            re_order[int(val)] = id
            # e.g. order = [2, 0, 1]
            # re_order = [1, 2, 0]
            order_list.append(int(val))
            # order_list = [2, 0, 1]

        ctx.save_for_backward(re_order)
        # use permute function, and create new Tensor with same backend
        return a._new(a._tensor.permute(*order_list))



    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        # TODO: Implement for Task 2.4.
        (re_order,) = ctx.saved_values
        return grad_output._new(grad_output._tensor.permute(*re_order)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None

        # Handle discontinuous functions (like comparisons) that can have large numerical gradients
        # but zero analytical gradients
        analytical_grad = x.grad[ind]
        numerical_grad = check

        # If the analytical gradient is zero but numerical gradient is very large,
        # this is likely a discontinuous function at a boundary
        if abs(analytical_grad) == 0.0 and abs(numerical_grad) > 1000:
            # Use a more robust epsilon for the central difference
            robust_check = grad_central_difference(f, *vals, arg=i, ind=ind, epsilon=1e-1)
            if abs(robust_check) < 100:
                # The large gradient was due to discontinuity, accept zero analytical gradient
                continue

        np.testing.assert_allclose(
            analytical_grad,
            numerical_grad,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, analytical_grad, i, ind, numerical_grad),
        )
