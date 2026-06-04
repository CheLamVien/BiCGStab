# BiCGStab for NGSolve

A small, standalone implementation of the preconditioned **BiCGStab** iterative solver for [NGSolve](https://ngsolve.org/). The solver is useful for nonsymmetric linear systems and follows NGSolve's `BaseMatrix` solver style.

## Features

- Preconditioned BiCGStab for systems of the form `A x = b`
- Optional `freedofs` projection for constrained degrees of freedom
- Optional residual printing and callbacks
- Residual history through `solver.residuals`
- Basic numerical breakdown diagnostics through `solver.breakdown_reason`
- Convenience function `BiCGStab(...)` for quick solves

## Requirements

- Python 3.9+
- NGSolve / Netgen

Install NGSolve using the method recommended for your platform. For many Python environments, this is:

```bash
pip install ngsolve
```

## Installation

Copy `bicgstab_ngsolve.py` into your project, or clone this repository and import it directly:

```python
from bicgstab_ngsolve import BiCGStab, BiCGStabSolver
```

## Quick usage

```python
from bicgstab_ngsolve import BiCGStab

# a.mat: assembled NGSolve matrix
# f.vec: right-hand-side vector
# fes: finite element space
# pre: optional NGSolve preconditioner

u_vec = BiCGStab(
    mat=a.mat,
    rhs=f.vec,
    freedofs=fes.FreeDofs(),
    pre=pre,
    tol=1e-10,
    maxsteps=500,
    printrates=True,
)
```

## Solver object usage

Use `BiCGStabSolver` directly when you want iteration metadata:

```python
from bicgstab_ngsolve import BiCGStabSolver

solver = BiCGStabSolver(
    mat=a.mat,
    pre=pre,
    freedofs=fes.FreeDofs(),
    tol=1e-10,
    maxiter=500,
    printrates="\r",
)

x = solver.Solve(f.vec)

print("iterations:", solver.iterations)
print("final residual:", solver.residuals[-1])
print("converged:", solver.converged)
print("breakdown:", solver.breakdown_reason)
```

## API

### `BiCGStab(...)`

Convenience wrapper that returns the solution vector.

```python
BiCGStab(
    mat,
    rhs,
    freedofs=None,
    pre=None,
    sol=None,
    tol=1e-12,
    maxsteps=100,
    printrates=True,
    initialize=True,
    conjugate=False,
    callback=None,
    **kwargs,
)
```

### `BiCGStabSolver`

`BaseMatrix`-compatible solver class.

Important attributes after `Solve(...)`:

- `solver.residuals`: residual norm history
- `solver.iterations`: number of residual checks
- `solver.converged`: whether the tolerance was reached
- `solver.breakdown_reason`: reason for early numerical breakdown, if any

## Notes

- `pre` should be an NGSolve preconditioner or matrix-like object that supports multiplication with a vector.
- `freedofs=fes.FreeDofs()` is recommended when solving finite element systems with constrained degrees of freedom.
- For complex-valued problems, set `conjugate=True` if you need the complex Hermitian inner product.

## License

Choose a license before publishing. MIT is a common choice for small numerical utility repositories.
