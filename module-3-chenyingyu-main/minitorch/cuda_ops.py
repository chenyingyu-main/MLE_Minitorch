# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device execution."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA or CPU execution."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32  # origin  = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """See `tensor_ops.py`"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.

        if i < out_size:
            # broadcast
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            # broadcast
            to_index(i, out_shape, out_index)

            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    Returns:
    -------
        None : Fills in `out`

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    # shared memory for this block

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # i = global thread index, = cuda.grid(1)
    pos = cuda.threadIdx.x
    # pos = local thread index in the block

    # TODO: Implement for Task 3.3.
    # Step 1: load into shared memory
    if i < size:
        cache[pos] = float(a[i])
    else:
        cache[pos] = 0.0

    # Step 2: make sure all threads have loaded
    cuda.syncthreads()

    if i < size:
        # All threads read from fast shared memory
        # Tree reduction: cutting in half the number of active threads
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                # Operation using the shared memory
                cache[pos] = cache[pos] + cache[pos + stride]
            cuda.syncthreads()
            stride //= 2

        # Step 3: write the result for this block to global memory
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """A practice sum function to prepare for reduce."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        # out position this block is computing

        pos = cuda.threadIdx.x
        # pos = local thread index in the block

        # TODO: Implement for Task 3.3.

        # make sure we are in bounds
        if out_pos < out_size:
            # broadcasting
            to_index(out_pos, out_shape, out_index)

            # size of the dimension (how many elements to reduce)
            reduce_size = a_shape[reduce_dim]
            reduce_stride = a_strides[reduce_dim]

            # every thread loads one or more elements
            # initialize reduce_value (like 0 for sum, 1 for mul)
            cache[pos] = reduce_value

            # [diff from CPU] every thread process some positions in reduce_dim
            # pos, pos + BLOCK_DIM, pos + 2*BLOCK_DIM, ...
            idx = pos
            while idx < reduce_size:
                # map the position for a to out_index (starting point)
                a_pos = index_to_position(out_index, a_strides)
                # calculate the position in a_storage to read
                current_pos = a_pos + idx * reduce_stride
                cache[pos] = fn(cache[pos], a_storage[current_pos])
                idx += BLOCK_DIM

            # make sure all threads have loaded and finish reduction
            cuda.syncthreads()

            # Tree Reduction in Shared Memory like sum_practice
            stride = BLOCK_DIM // 2
            while stride > 0:
                if pos < stride:
                    cache[pos] = fn(cache[pos], cache[pos + stride])
                cuda.syncthreads()
                stride //= 2

            if pos == 0:
                out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # Allocate shared memory tiles for a and b
    tile_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    tile_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i, j = cuda.grid(2)

    # load from global memory to shared memory (tile_a and tile_b)
    if i < size and j < size:
        tile_a[tx, ty] = a[i * size + j]
        tile_b[tx, ty] = b[i * size + j]
    else:
        tile_a[tx, ty] = 0.0
        tile_b[tx, ty] = 0.0
    cuda.syncthreads()

    # compute from shared memory (tile)
    if i < size and j < size:
        accumu = 0.0
        for k in range(size):
            accumu += tile_a[tx, k] * tile_b[k, ty]

        out[i * size + j] = accumu


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """A practice matrix multiply function"""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.

    # a_shape: [batch, M, K]
    # b_shape: [batch, K, N]
    # out_shape: [batch, M, N]
    M = a_shape[1]
    K = a_shape[2]
    N = b_shape[2]

    # Accumulator for the dot product
    accum = 0.0

    # how many tiles we need to cover K dimension
    # ex: K = 128, BLOCK_DIM = 32
    # num_tiles = (128 + 31) // 32 = 4 we need 4 tiles to cover 128 dim
    num_tiles = (K + BLOCK_DIM - 1) // BLOCK_DIM

    # Loop over tiles along the K dimension
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * BLOCK_DIM

        # Move across shared dimension by block dim.
        # a) Copy into shared memory for a matrix.
        a_row = i
        a_col = tile_start + pj

        # check bounds before loading
        if a_row < M and a_col < K:
            a_pos = batch * a_batch_stride + a_row * a_strides[1] + a_col * a_strides[2]
            a_shared[pi, pj] = a_storage[a_pos]
        else:
            a_shared[pi, pj] = 0.0

        # b) Copy into shared memory for b matrix
        b_row = tile_start + pi
        b_col = j
        if b_row < K and b_col < N:
            b_pos = batch * b_batch_stride + b_row * b_strides[1] + b_col * b_strides[2]
            b_shared[pi, pj] = b_storage[b_pos]
        else:
            b_shared[pi, pj] = 0.0

        # sync
        cuda.syncthreads()

        # c) Compute the dot produce for position c[i, j]
        for k in range(BLOCK_DIM):
            accum += a_shared[pi, k] * b_shared[k, pj]

        cuda.syncthreads()

    # Write the result to global memory
    if i < M and j < N:
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_pos] = accum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
