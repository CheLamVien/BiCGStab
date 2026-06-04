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
