from __future__ import annotations

from typing import TYPE_CHECKING
from abc import abstractmethod

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls: type, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls: type, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls: type, *vals: "ScalarLike") -> Scalar:
        """Apply the function to the given Scalar inputs."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)

    @staticmethod
    @abstractmethod
    def forward(ctx: Context, *args: float) -> float:
        """Computes the forward pass of the function.

        Args:
        ----
            ctx (Context): Context object to save information for backward computation.
            *args (float): Input values.

        Returns:
        -------
            float: The result of the forward computation.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Forward method not implemented.")

    @staticmethod
    @abstractmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass (derivative) of the function.

        Args:
        ----
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Derivative of the output with respect to some scalar.

        Returns:
        -------
            Tuple[float, ...]: The gradients with respect to each input.

        Raises:
        ------
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Backward method not implemented.")


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of the input values a and b."""
        return a + b  # operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradients of the addition with respect to its inputs."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of the input value a."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the logarithm with respect to its input."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


### To implement for Task 1.2 and 1.4 ###
# Look at the above classes for examples on how to implement the forward and backward functions
# Use the operators.py file from Module 0


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the product of the input values a and b."""
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradients of the multiplication with respect to its inputs."""
        # TODO: Implement for Task 1.4.
        (a, b) = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse (reciprocal) of the input value a."""
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the inverse with respect to its input."""
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the negation of the input value a."""
        # TODO: Implement for Task 1.2.
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the negation with respect to its input."""
        # TODO: Implement for Task 1.4.
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of the input value a."""
        # TODO: Implement for Task 1.2.
        # Note: the derivative formula uses σ(a) * (1 - σ(a))
        # so, for sigmoid forward, we save the output (σ(a)) instead of the input :)
        sigmoid_val = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the sigmoid with respect to its input.

        Derivative of sigmoid:
        dσ/da = σ(a) * (1 - σ(a))
        s = σ(a)
        So chain rule: dL/da = dL/dσ * dσ/da
                             = d_output * s * (1 - s)
        """
        # TODO: Implement for Task 1.4.

        # Derivative of sigmoid:
        # dσ/da = σ(a) * (1 - σ(a))
        # s = σ(a)
        # So chain rule: dL/da = dL/dσ * dσ/da
        #                      = d_output * s * (1 - s)

        (sigmoid_val,) = ctx.saved_values
        return operators.mul(operators.mul(sigmoid_val, (1 - sigmoid_val)), d_output)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU (Rectified Linear Unit) of the input value a."""
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the ReLU with respect to its input."""
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of the input value a."""
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the exponential with respect to its input.

        The derivative of exp(a) is exp(a), so the gradient is:
        dL/da = dL/d(exp(a)) * exp(a) = d_output * exp(a)
        """
        # TODO: Implement for Task 1.4.
        # Derivative of exp:
        # d(exp(a))/da = exp(a)
        # So chain rule: dL/da = dL/d(exp(a)) * d
        #                      = d_output * exp(a)

        (a,) = ctx.saved_values
        return operators.mul(operators.exp(a), d_output)


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the less-than indicator as a float."""
        # TODO: Implement for Task 1.2.
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Less-than is non-differentiable and treated as having zero gradient almost everywhere,
        so the upstream gradient is ignored.
        """
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the equality indicator as a float."""
        # TODO: Implement for Task 1.2.
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Equality is non-differentiable and treated as having zero gradient almost everywhere,
        so the upstream gradient is ignored.
        """
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0
