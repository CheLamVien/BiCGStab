"""BiCGStab solver for NGSolve.

This module provides a small BaseMatrix-compatible implementation of the
preconditioned BiCGStab method. It follows the style of NGSolve's iterative
solvers and can be used either through the ``BiCGStabSolver`` class or the
convenience function ``BiCGStab``.
"""

from __future__ import annotations

import os
from math import log
from typing import Callable, Optional, Union

from netgen.libngpy._meshing import _GetStatus, _PushStatus, _SetThreadPercentage
from ngsolve import BaseMatrix, BaseVector, BitArray, Norm, Preconditioner, Projector, TimeFunction


_CLEAR_LINE = "" if os.name == "nt" else "\33[2K"
_BREAKDOWN_TOL = 1e-30


_LINEAR_SOLVER_DOC = """
Parameters
----------
mat
    Matrix/operator for the left-hand side ``A`` in ``A x = b``.
pre
    Optional preconditioner/operator approximating ``A^{-1}``.
freedofs
    Optional NGSolve ``BitArray`` of free degrees of freedom. When provided,
    residuals and matrix-vector products are projected to the free subspace.
tol
    Relative residual tolerance. If ``atol`` is also provided, convergence is
    reached when the residual is below ``max(tol * ||r0||, atol)``.
maxiter
    Maximum number of residual checks/iterations.
atol
    Absolute residual tolerance.
callback
    Optional function called as ``callback(iteration, residual)``.
callback_sol
    Optional function called with the current solution vector.
printrates
    Print residual history. Pass ``"\\r"`` for one-line updating output.
"""


class LinearSolver(BaseMatrix):
    """Small BaseMatrix-compatible base class for iterative solvers.""" + _LINEAR_SOLVER_DOC

    name = "LinearSolver"

    def __init__(
        self,
        mat: BaseMatrix,
        pre: Optional[Union[Preconditioner, BaseMatrix]] = None,
        freedofs: Optional[BitArray] = None,
        tol: Optional[float] = None,
        maxiter: int = 100,
        atol: Optional[float] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        callback_sol: Optional[Callable[[BaseVector], None]] = None,
        printrates: Union[bool, str] = False,
    ) -> None:
        super().__init__()

        if tol is None and atol is None:
            tol = 1e-12

        self.mat = mat
        self.pre = pre
        self.freedofs = freedofs
        self.tol = tol
        self.atol = atol
        self.maxiter = maxiter
        self.callback = callback
        self.callback_sol = callback_sol
        self.printrates = printrates

        self.sol: Optional[BaseVector] = None
        self.residuals: list[float] = []
        self.iterations = 0
        self.converged = False
        self.breakdown_reason: Optional[str] = None
        self._target_residual: Optional[float] = None

    @TimeFunction
    def Solve(
        self,
        rhs: BaseVector,
        sol: Optional[BaseVector] = None,
        initialize: bool = True,
    ) -> BaseVector:
        """Solve ``A sol = rhs`` and return the solution vector."""
        self.iterations = 0
        self.residuals = []
        self.converged = False
        self.breakdown_reason = None
        self._target_residual = None

        old_status = _GetStatus()
        _PushStatus(f"{self.name} Solve")
        _SetThreadPercentage(0)

        try:
            if sol is None:
                sol = rhs.CreateVector()
                initialize = True
            if initialize:
                sol[:] = 0

            self.sol = sol
            self._SolveImpl(rhs=rhs, sol=sol)
            return sol
        finally:
            if old_status[0] != "idle":
                _PushStatus(old_status[0])
                _SetThreadPercentage(old_status[1])

    def Height(self) -> int:
        return self.mat.height

    def Width(self) -> int:
        return self.mat.width

    def CreateVector(self, col: bool) -> BaseVector:
        return self.mat.CreateVector(col)

    def IsComplex(self) -> bool:
        return self.mat.IsComplex()

    def Mult(self, x: BaseVector, y: BaseVector) -> None:
        self.Solve(rhs=x, sol=y, initialize=True)

    def Update(self) -> None:
        if self.pre is not None and hasattr(self.pre, "Update"):
            self.pre.Update()

    def CheckResidual(self, residual: float) -> bool:
        """Record a residual and return ``True`` when the stop criterion is met."""
        self.iterations += 1
        self.residuals.append(residual)

        if len(self.residuals) == 1:
            if self.tol is None:
                self._target_residual = self.atol
            else:
                self._target_residual = residual * self.tol
                if self.atol is not None:
                    self._target_residual = max(self._target_residual, self.atol)
        else:
            if self.callback is not None:
                self.callback(self.iterations, residual)
            if self.callback_sol is not None and self.sol is not None:
                self.callback_sol(self.sol)

        target = self._target_residual or 0.0
        self._update_ngsolve_progress(residual, target)
        self._print_rate(residual, target)

        self.converged = residual <= target
        stopped = self.converged or self.iterations >= self.maxiter

        if stopped and self.printrates == "\r":
            status = "converged" if self.converged else "NOT converged"
            print(
                f"{_CLEAR_LINE}{self.name} {status} in "
                f"{self.iterations} iterations to residual {residual}"
            )
        return stopped

    def _update_ngsolve_progress(self, residual: float, target: float) -> None:
        first = self.residuals[0]
        if first <= 0 or target <= 0:
            return

        if residual <= 0:
            _SetThreadPercentage(100)
            return

        try:
            progress_from_residual = (log(residual) - log(first)) / (log(target) - log(first))
        except (ValueError, ZeroDivisionError):
            progress_from_residual = 0.0

        progress = 100.0 * max(self.iterations / self.maxiter, progress_from_residual)
        _SetThreadPercentage(min(100.0, max(0.0, progress)))

    def _print_rate(self, residual: float, target: float) -> None:
        if not self.printrates:
            return

        end = "\n" if isinstance(self.printrates, bool) else self.printrates
        print(
            f"{_CLEAR_LINE}{self.name} iteration {self.iterations}, "
            f"residual = {residual}     ",
            end=end,
        )
        if self.iterations == self.maxiter and residual > target:
            print(f"{_CLEAR_LINE}WARNING: {self.name} did not converge to TOL")


class BiCGStabSolver(LinearSolver):
    """Preconditioned BiCGStab solver for nonsymmetric linear systems.""" + _LINEAR_SOLVER_DOC + """
conjugate
    Passed to NGSolve's ``InnerProduct``. Use ``True`` for the complex Hermitian
    inner product; keep ``False`` for the pseudo inner product used by NGSolve
    for some complex symmetric operators.
"""

    name = "BiCGStab"

    def __init__(
        self,
        *args,
        conjugate: bool = False,
        abstol: Optional[float] = None,
        maxsteps: Optional[int] = None,
        printing: Union[bool, str] = False,
        **kwargs,
    ) -> None:
        # Backward-compatible aliases used by some NGSolve solvers.
        if printing:
            print("WARNING: printing is deprecated, use printrates instead")
            kwargs["printrates"] = printing
        if abstol is not None:
            print("WARNING: abstol is deprecated, use atol instead")
            kwargs["atol"] = abstol
        if maxsteps is not None:
            print("WARNING: maxsteps is deprecated, use maxiter instead")
            kwargs["maxiter"] = maxsteps

        super().__init__(*args, **kwargs)
        self.conjugate = conjugate

    @property
    def errors(self) -> list[float]:
        """Backward-compatible alias for ``residuals``."""
        return self.residuals

    def _SolveImpl(self, rhs: BaseVector, sol: BaseVector) -> None:
        r, r_hat, p, p_hat, s, s_hat, t, v = [sol.CreateVector() for _ in range(8)]
        projector = Projector(self.freedofs, True) if self.freedofs is not None else None
        conjugate = self.conjugate

        r.data = rhs - self.mat * sol
        self._project_inplace(r, projector)

        r_hat.data = r
        p.data = r
        rho = r_hat.InnerProduct(r, conjugate=conjugate)

        if self.CheckResidual(Norm(r)):
            return

        for _ in range(1, self.maxiter + 1):
            self._apply_preconditioner(p, p_hat)

            v.data = self.mat * p_hat
            self._project_inplace(v, projector)

            alpha_den = r_hat.InnerProduct(v, conjugate=conjugate)
            if abs(alpha_den) < _BREAKDOWN_TOL:
                self.breakdown_reason = "alpha denominator is numerically zero"
                return
            alpha = rho / alpha_den

            s.data = r
            s.data -= alpha * v
            self._project_inplace(s, projector)

            if self.CheckResidual(Norm(s)):
                sol.data += alpha * p_hat
                return

            self._apply_preconditioner(s, s_hat)
            t.data = self.mat * s_hat
            self._project_inplace(t, projector)

            omega_den = t.InnerProduct(t, conjugate=conjugate)
            if abs(omega_den) < _BREAKDOWN_TOL:
                self.breakdown_reason = "omega denominator is numerically zero"
                return
            omega = t.InnerProduct(s, conjugate=conjugate) / omega_den

            sol.data += alpha * p_hat + omega * s_hat

            r.data = s
            r.data -= omega * t
            self._project_inplace(r, projector)

            if self.CheckResidual(Norm(r)):
                return

            rho_old = rho
            rho = r_hat.InnerProduct(r, conjugate=conjugate)
            if abs(rho_old) < _BREAKDOWN_TOL:
                self.breakdown_reason = "rho is numerically zero"
                return
            if abs(omega) < _BREAKDOWN_TOL:
                self.breakdown_reason = "omega is numerically zero"
                return

            beta = (rho / rho_old) * (alpha / omega)
            p.data = r
            p.data += beta * (p - omega * v)

    def _apply_preconditioner(self, x: BaseVector, y: BaseVector) -> None:
        if self.pre is None:
            y.data = x
        else:
            y.data = self.pre * x

    @staticmethod
    def _project_inplace(vec: BaseVector, projector: Optional[Projector]) -> None:
        if projector is not None:
            vec.data = projector * vec


def BiCGStab(
    mat: BaseMatrix,
    rhs: BaseVector,
    freedofs: Optional[BitArray] = None,
    pre: Optional[Union[Preconditioner, BaseMatrix]] = None,
    sol: Optional[BaseVector] = None,
    tol: float = 1e-12,
    maxsteps: int = 100,
    printrates: Union[bool, str] = True,
    initialize: bool = True,
    conjugate: bool = False,
    callback: Optional[Callable[[int, float], None]] = None,
    **kwargs,
) -> BaseVector:
    """Solve ``mat * sol = rhs`` with preconditioned BiCGStab.

    This convenience wrapper returns only the solution vector. Use
    ``BiCGStabSolver`` directly when you also need residual history,
    iteration count, or breakdown diagnostics.
    """
    solver = BiCGStabSolver(
        mat=mat,
        freedofs=freedofs,
        pre=pre,
        conjugate=conjugate,
        tol=tol,
        maxiter=maxsteps,
        callback=callback,
        printrates=printrates,
        **kwargs,
    )
    return solver.Solve(rhs=rhs, sol=sol, initialize=initialize)
