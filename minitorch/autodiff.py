from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant (no `derivative`)"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for backpropagation."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    top_order: List[Variable] = []

    def dfs(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    dfs(parent)
        visited.add(var.unique_id)
        top_order.insert(0, var)

    dfs(variable)
    return top_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    r"""Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: The derivative of the output with respect to `variable`

    Returns:
    -------
        No return. Should write its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    toporder = topological_sort(variable)
    derivs = {}
    derivs[variable.unique_id] = deriv

    for var in toporder:
        deriv = derivs[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for parent, localderiv in var.chain_rule(deriv):
                if parent.is_constant():
                    continue
                derivs.setdefault(parent.unique_id, 0.0)
                derivs[parent.unique_id] = derivs[parent.unique_id] + localderiv


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
        """Return the saved tensors."""
        return self.saved_values
