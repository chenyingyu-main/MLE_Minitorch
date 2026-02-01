from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    Uses the central difference formula: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant for finite difference approximation

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals_list = list(vals)
    x1 = vals_list.copy()
    x2 = vals_list.copy()
    x1[arg] += epsilon
    x2[arg] -= epsilon

    return (f(*x1) - f(*x2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable."""
        pass

    @property
    def unique_id(self) -> int:
        """A unique identifier for this variable."""
        pass

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        pass

    def is_constant(self) -> bool:
        """True if this variable is constant (no history)"""
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables of this variable."""
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Apply the chain rule to get the derivatives of this variable's parents."""
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.

    Hints:
        - Use depth-first search (DFS) to visit nodes
        - Track visited nodes to avoid cycles (use node.unique_id)
        - Return nodes in reverse order (dependencies first)

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    order: Iterable[Variable] = []  # output list: order of nodes in topological order

    def dfs(node: Variable) -> None:
        # if visited return, if constant, we dont deal with it
        if node.unique_id in visited or node.is_constant():
            return

        visited.add(node.unique_id)
        # if is_leaf, therse is no parent, just pass, and add itself to order
        # if node.is_leaf():
        #     pass

        if not node.is_leaf():
            for parent in node.parents:
                dfs(parent)

        order.append(node)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:
        - First get all nodes in topological order using topological_sort()
        - Create a dictionary to store derivatives for each node (keyed by unique_id)
        - Initialize the starting node's derivative to the input deriv
        - Process nodes in the topological order (which is already correct for backprop)
        - For leaf nodes: call node.accumulate_derivative(derivative)
        - For non-leaf nodes: call node.chain_rule(derivative) to get parent derivatives
        - Sum derivatives when the same parent appears multiple times

    """
    # TODO: Implement for Task 1.4.

    # 1. First get all nodes in topological order using topological_sort()
    nodes_in_order: Iterable[Variable] = topological_sort(variable)

    # 2. Create a dictionary to store derivatives for each node (keyed by unique_id)
    derivative_dict = {}  # key: var.unique_id, value: derivative
    # Initialize the starting node's derivative to the input deriv
    derivative_dict[variable.unique_id] = deriv

    # 3. Process nodes in the topological order (which is already correct for backprop)
    for node in nodes_in_order:
        derivative = derivative_dict.get(node.unique_id, 0.0)
        if node.is_leaf():
            # 3-1: For leaf nodes: call node.accumulate_derivative(derivative)
            node.accumulate_derivative(derivative)
        else:
            # 3-2: For non-leaf nodes: call node.chain_rule(derivative) to get parent derivatives
            parent_derivatives = node.chain_rule(derivative)
            for parent, parent_deriv in parent_derivatives:
                if parent.unique_id in derivative_dict:
                    # if the same parent appears multiple times (already in the dict), sum the derivatives
                    derivative_dict[parent.unique_id] += parent_deriv
                else:
                    derivative_dict[parent.unique_id] = parent_deriv

    # """
    # ===============================================================================
    # Example: Backpropagation with Multiple Paths
    # ===============================================================================

    # Computation Graph:
    # ------------------
    #         x (leaf, value=2.0)
    #         │
    #         ├──────────┐
    #         │          │
    #         ▼          ▼
    #     w = x + 3    v = x * 2
    #     (value=5.0)  (value=4.0)
    #         │          │
    #         └────┬─────┘
    #             ▼
    #         z = w * v
    #         (value=20.0)

    # Goal: Compute dz/dx using backpropagation
    # Starting with dz/dz = 1.0

    # ===============================================================================
    # Step-by-Step Execution:
    # ===============================================================================

    # Initial State:
    # --------------
    # nodes_in_order = [z, w, v, x]  (from topological_sort)
    # derivative_dict = {z.id: 1.0}

    # -------------------------------------------------------------------------------
    # Iteration 1: Process z (non-leaf)
    # -------------------------------------------------------------------------------
    # Current node: z (z = w * v, value=20.0)
    # derivative = derivative_dict[z.id] = 1.0

    # z.chain_rule(1.0) returns:
    # - (w, dz/dw * 1.0) = (w, v * 1.0) = (w, 4.0)
    # - (v, dz/dv * 1.0) = (v, w * 1.0) = (v, 5.0)

    # Update derivative_dict:
    # derivative_dict[w.id] = 4.0
    # derivative_dict[v.id] = 5.0

    # Current state:
    # derivative_dict = {z.id: 1.0, w.id: 4.0, v.id: 5.0}

    # -------------------------------------------------------------------------------
    # Iteration 2: Process w (non-leaf)
    # -------------------------------------------------------------------------------
    # Current node: w (w = x + 3, value=5.0)
    # derivative = derivative_dict[w.id] = 4.0

    # w.chain_rule(4.0) returns:
    # - (x, dw/dx * 4.0) = (x, 1.0 * 4.0) = (x, 4.0)

    # Update derivative_dict:
    # derivative_dict[x.id] = 4.0  (first time seeing x)

    # Current state:
    # derivative_dict = {z.id: 1.0, w.id: 4.0, v.id: 5.0, x.id: 4.0}

    # -------------------------------------------------------------------------------
    # Iteration 3: Process v (non-leaf)
    # -------------------------------------------------------------------------------
    # Current node: v (v = x * 2, value=4.0)
    # derivative = derivative_dict[v.id] = 5.0

    # v.chain_rule(5.0) returns:
    # - (x, dv/dx * 5.0) = (x, 2.0 * 5.0) = (x, 10.0)

    # Update derivative_dict:
    # x.id already in dict! Accumulate:
    # derivative_dict[x.id] += 10.0  →  4.0 + 10.0 = 14.0

    # Current state:
    # derivative_dict = {z.id: 1.0, w.id: 4.0, v.id: 5.0, x.id: 14.0}

    # -------------------------------------------------------------------------------
    # Iteration 4: Process x (leaf)
    # -------------------------------------------------------------------------------
    # Current node: x (leaf, value=2.0)
    # derivative = derivative_dict[x.id] = 14.0

    # x.is_leaf() = True
    # Call: x.accumulate_derivative(14.0)

    # Final result: x.derivative = 14.0

    # ===============================================================================
    # Verification:
    # ===============================================================================

    # Chain rule calculation:
    # dz/dx = (dz/dw * dw/dx) + (dz/dv * dv/dx)
    #     = (v * 1) + (w * 2)
    #     = (4.0 * 1) + (5.0 * 2)
    #     = 4.0 + 10.0
    #     = 14.0 ✓

    # Gradient Flow Diagram:
    # ---------------------
    #             z (grad=1.0)
    #             │
    #         ┌─────┴─────┐
    #         │           │
    #     dz/dw=4.0   dz/dv=5.0
    #         │           │
    #         ▼           ▼
    #     w (grad=4.0) v (grad=5.0)
    #         │           │
    #     dw/dx=1.0   dv/dx=2.0
    #         │           │
    #         └─────┬─────┘
    #             │ (accumulate)
    #             ▼
    #         x (grad=14.0)

    # Key Insight:
    # ------------
    # The variable x appears in TWO paths to z:
    # Path 1: x → w → z  (contributes 4.0)
    # Path 2: x → v → z  (contributes 10.0)

    # The backpropagation algorithm correctly accumulates gradients from both paths,
    # giving the total derivative dz/dx = 14.0.

    # ===============================================================================
    # """


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve the saved values for backward computation."""
        return self.saved_values
