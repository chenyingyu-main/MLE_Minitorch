from typing import Tuple

from .tensor import Tensor
from .tensor_functions import Function, rand
from . import operators
from .autodiff import Context
from .fast_ops import FastOps


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.

    new_height = height // kh
    new_width = width // kw

    # # Example
    # shape: (1, 1, 4, 4)
    # batch=1, channel=1, height=4, width=4
    # kernel = (2, 2)  # kh=2, kw=2
    #
    # (batch=0, channel=0):
    # [
    #     [1,  2,  3,  4],   ← row 0
    #     [5,  6,  7,  8],   ← row 1
    #     [9,  10, 11, 12],  ← row 2
    #     [13, 14, 15, 16]   ← row 3
    # ]

    # cut from height and width
    # Step 1: cut from the height (batch, channel, new_height, kh, width)
    reshaped = input.contiguous().view(batch, channel, new_height, kh, width)

    # After step 1 (batch=0, channel=0):
    # [
    #     # new_height=0 (first 2 row)
    #     [
    #         [1,  2,  3,  4],   ← kh=0
    #         [5,  6,  7,  8]    ← kh=1
    #     ],
    #     # new_height=1 (last 2 row)
    #     [
    #         [9,  10, 11, 12],  ← kh=0
    #         [13, 14, 15, 16]   ← kh=1
    #     ]
    # ]
    # shape = (1, 1, 2, 2, 4) -> batch=1, channel=1, new_height=2, kh=2, width=4

    # Step 2: cut from width (batch, channel, new_height, kh, new_width, kw)
    reshaped = reshaped.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # After step 2 (batch=0, channel=0):
    # [
    #     # new_height=0
    #     [
    #         # kh=0
    #         [
    #             [1, 2],    # new_width=0, kw=[0,1]
    #             [3, 4]     # new_width=1, kw=[0,1]
    #         ],
    #         # kh=1
    #         [
    #             [5, 6],    # new_width=0, kw=[0,1]
    #             [7, 8]     # new_width=1, kw=[0,1]
    #         ]
    #     ],
    #     # new_height=1
    #     [
    #         # kh=0
    #         [
    #             [9,  10],  # new_width=0, kw=[0,1]
    #             [11, 12]   # new_width=1, kw=[0,1]
    #         ],
    #         # kh=1
    #         [
    #             [13, 14],  # new_width=0, kw=[0,1]
    #             [15, 16]   # new_width=1, kw=[0,1]
    #         ]
    #     ]
    # ]

    # shape = (1, 1, 2, 2, 2, 2)

    # Step 3: reorder (batch, channel, new_height, new_width, kh, kw)
    reshaped = reshaped.permute(0, 1, 2, 4, 3, 5)

    # After step 3 (batch=0, channel=0):
    # [
    #     # new_height=0
    #     [
    #         # new_width=0
    #         [
    #             # kh=0, kw=0,1
    #             [1, 2],
    #             # kh=1, kw=0,1
    #             [5, 6]
    #         ],
    #         # new_width=1
    #         [
    #             # kh=0, kw=0,1
    #             [3, 4],
    #             # kh=1, kw=0,1
    #             [7, 8]
    #         ]
    #     ],
    #     # new_height=1
    #    [............]

    # Step 4: combine kernel dimension (batch, channel, new_height, new_width, kh*kw)
    reshaped = reshaped.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )

    # flatten the kh and kw dimensions (into one dimension)
    # After step 4 (batch=0, channel=0):
    # [
    #     # new_height=0
    #     [
    #         # new_width=0
    #         [1, 2, 5, 6],      # left top 2x2 block flattened
    #         # new_width=1
    #         [3, 4, 7, 8]       # right top 2x2 block flattened
    #     ],
    #     # new_height=1
    #     [......]

    return reshaped, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    # TODO: Implement for Task 4.3.
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)

    # calculate the avg for the last dimension (kh*kw)
    pooled = tiled.mean(dim=4)  # dim =4 mean over the last dimension
    # pooled shape: (batch, channel, new_height, new_width)

    # Step 3: view, shape to the right shape
    pooled = pooled.view(batch, channel, new_height, new_width)

    return pooled


max_reduce = FastOps.reduce(operators.max, -1e9)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction

        Mathematical operation:
            output[i] = max(input[i, :]) along dimension dim

        Example:
            input = [[1, 5, 3],    dim = 1
                     [2, 4, 6]]
            output = [5, 6]

        """
        # TODO: Implement for Task 4.4.
        dim_val = int(dim.item())
        max_red = max_reduce(a, dim_val)
        # Save input, max result, and dimension
        ctx.save_for_backward(a, max_red, dim_val)

        return max_red

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward of max should be argmax (see argmax function)

        Mathematical operation:
            ∂L/∂x_i = ∂L/∂y  if x_i == max(x)
            ∂L/∂x_i = 0      otherwise

        Gradient only flows to the maximum value position.
        """
        # TODO: Implement for Task 4.4.
        (a, max_red, dim_val) = ctx.saved_values

        # find the pos of the max
        is_max = max_red == a

        # Along the dim, count how many max we have
        count = is_max.sum(dim_val)

        # share the grad
        return (grad_output * is_max / count), grad_output.zeros()


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction over a dimension

    Mathematical operation:
        output = max(input, dim)

    Args:
        input: input tensor
        dim: dimension to reduce over

    Returns:
        tensor with max values along the specified dimension

    """
    # TODO: Implement for Task 4.4.
    return Max.apply(input, input._ensure_tensor(dim))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Mathematical operation:
        output[i] = 1  if input[i] == max(input)
        output[i] = 0  otherwise

    Example:
        input = [1, 5, 3]
        max = 5 (at index 1)
        argmax = [0, 1, 0]  (one-hot encoding)

    Implementation:
        1. Compute max along dimension
        2. Compare input == max (broadcasts automatically)
        3. Result is 1.0 where equal, 0.0 elsewhere

    """
    # TODO: Implement for Task 4.4.
    # Get max values along dimension
    max_vals = max(input, dim)

    # Returns 1.0 where input == max, 0.0 elsewhere
    return input == max_vals


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Mathematical formula (naive):
        softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    Mathematical formula (numerically stable):
        softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
        - Prevents overflow when x_i is large (e.g., x_i = 1000)

    Properties:
        - All outputs are in (0, 1)
        - Sum of outputs = 1.0 (probability distribution)
    """
    # TODO: Implement for Task 4.4.
    shifted = input - max(input, dim)
    t = shifted.exp()
    s = t.sum(dim)
    return t / s


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Mathematical formula (naive, unstable):
        logsoftmax(x_i) = log(softmax(x_i))
                        = log(exp(x_i) / Σ_j exp(x_j))

    Mathematical formula (stable, using LogSumExp trick):
        logsoftmax(x_i) = x_i - log(Σ_j exp(x_j))
                        = x_i - max(x) - log(Σ_j exp(x_j - max(x)))

    Derivation:
        log(exp(x_i) / Σ exp(x_j))
        = log(exp(x_i)) - log(Σ exp(x_j))
        = x_i - log(Σ exp(x_j))

    Do not compute log(softmax(x))
        - softmax(x) might be very small (e.g., 1e-40)
        - Direct computation avoids this intermediate step

    => For stability:
        log(Σ exp(x_j)) = log(Σ exp(x_j - c + c))
                        = log(exp(c) * Σ exp(x_j - c))
                        = c + log(Σ exp(x_j - c))

        Choose c = max(x) to prevent overflow in exp()

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    """
    # TODO: Implement for Task 4.4.
    shifted = input - max(input, dim)  # x_i - max(x)
    t = shifted.exp()
    t = t.sum(dim)
    t = t.log()  # log(Σ_j exp(x_j - max(x)))
    return shifted - t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    # TODO: Implement for Task 4.4.
    # Mathematical operation:
    #     For each (kh × kw) window in the input:
    #         output[i, j] = max(input[i:i+kh, j:j+kw])

    # Example (2×2 kernel):
    #     input (4×4):              output (2×2):
    #     ┌──────────────┐         ┌─────┐
    #     │ 1  2│ 3  4  │         │ 6  8│
    #     │ 5  6│ 7  8  │    →    │14 16│
    #     ├─────┼───────┤         └─────┘
    #     │ 9 10│11 12  │
    #     │13 14│15 16  │
    #     └──────────────┘

    #     Top-left 2×2: max(1,2,5,6) = 6
    #     Top-right 2×2: max(3,4,7,8) = 8
    #     Bottom-left 2×2: max(9,10,13,14) = 14
    #     Bottom-right 2×2: max(11,12,15,16) = 16

    # Implementation strategy:
    #     1. Use tile() to reshape input into pooling windows
    #     2. Apply max reduction over the window dimension

    # Args:
    # ----
    #     input: batch x channel x height x width
    #     kernel: height x width of pooling

    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)

    # Apply max over the last dimension (pooling window)
    # Result shape: (batch, channel, new_height, new_width)
    pooled = max(tiled, 4)

    pooled = pooled.view(batch, channel, new_height, new_width)

    return pooled


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise

    Args:
    ----
        input: input tensor
        rate: probability [0, 1) of dropping out each position
        ignore: skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with random positions dropped out

    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return input
    rand_tensor = rand(input.shape)
    random_drop = rand_tensor > rate

    return input * random_drop
