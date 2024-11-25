from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    Eq,
    Lt,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass(unsafe_hash=True)
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant (no `derivative`)"""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parents of this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for backpropagation.
        This method calculates the gradients of the output with respect to the
        input variables by applying the chain rule. It uses the history of
        operations stored in the `history` attribute to backpropagate the
        gradients through the computational graph.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to some
                    variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples where each
                            tuple contains a `Variable` and its
                            corresponding gradient.

        Raises:
        ------
            NotImplementedError: If the method is not yet implemented.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        localderivs = h.last_fn._backward(h.ctx, d_output)
        return list(zip(h.inputs, localderivs))

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __add__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, b)

    def __lt__(self, b: ScalarLike) -> Scalar:
        return Lt.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        return Lt.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        return Eq.apply(b, self)

    def __sub__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, -b)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def log(self) -> Scalar:
        """Apply the natural logarithm to the scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Apply the exponential function to the scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Apply the sigmoid function to the scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Apply the ReLU function to the scalar."""
        return ReLU.apply(self)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks the derivative of a function using central difference approximation.
    This function computes the derivative of the given function `f` at the
    provided scalar arguments using automatic differentiation and compares
    it against the derivative computed using the central difference method.

    Parameters
    ----------
    f : Any
        The function whose derivative is to be checked.
    *scalars : Scalar
        The scalar arguments at which the derivative of the function `f`
        is to be checked.

    Raises
    ------
    AssertionError
        If the derivative computed using automatic differentiation does not
        match the derivative computed using the central difference method
        within a tolerance of 1e-2.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
