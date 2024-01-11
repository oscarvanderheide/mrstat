# MR-STAT Reconstruction Code

This repository contains an implementation of the partially-matrix free, inexact Gauss-Newton algorithm that can be used for MR-STAT. It use a numerical brain phantom to generate synthetic data from which it then estimates $T_1$, $T_2$ and proton density. It only supports Cartesian readout trajectories at the moment. The reconstruction makes use of [BlochSimulators.jl](https://github.com/oscarvanderheide/BlochSimulators.jl) and should be able to run on modern NVIDIA GPU hardware.
