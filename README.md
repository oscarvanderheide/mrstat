# MR-STAT Reconstruction Code

This repository contains an implementation of the partially-matrix free, inexact Gauss-Newton algorithm that can be used for MR-STAT. It use a 2D numerical brain phantom to generate synthetic data from which it then estimates $T_1$, $T_2$ and proton density. It only supports Cartesian readout trajectories at the moment. The reconstruction makes use of [BlochSimulators.jl](https://github.com/oscarvanderheide/BlochSimulators.jl) and should be able to run on modern NVIDIA GPU hardware.

## Warning:
- This codebase uses an older version of BlochSimulators (v0.2.7), make sure to `instantiate` the environment as defined in the manifest.
- This codebase is structured as a (non-registered) Julia package. 
- This codebase assumes a CUDA device is available.

## Structure:
- Only the `MRSTAT.main` needs be run. It loads external packages and several functions, assembles a FISP sequence, a Cartesian trajectory and a numerical phantom (including spatial coordinates and coil sensitivity profiles). It generates (noiseless) data. An MR-STAT reconstruction is then performed on the generated data.
- The objective function is defined in `utils/objective.jl`. This function is also used by the non-linear solver to calculate the gradient of the objective and the (Gauss-Newton approximation to) the Hessian.
- Bloch simulations (in case of FISP: EPG simulations) are performed using the `BlochSimulators.jl` package. 
- Partial derivatives are calculated using a finite difference approach, see `DerivativeOperations/simulate_derivatives.jl`.
- The matrix-vector products with the Jacobian matrix and its adjoint are implemented in a partially matrix-free method, see `DerivativeOperations/Jv.jl` and `DerivativeOperations/Jᴴv.jl`.
- The non-linear solver (Trust Region Reflective with Gauss-Newton approximation to the Hessian) is contained in the `TrustRegionReflective` submodule.
