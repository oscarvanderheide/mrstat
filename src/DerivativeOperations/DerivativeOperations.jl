module DerivativeOperations

    using StaticArrays
    using ComputationalResources
    # using Adapt
    using CUDA
    using LinearAlgebra
    using InteractiveUtils
    using StructArrays

    using BlochSimulators

    include("types.jl")
    include("simulate_derivatives.jl")

    include("Jv.jl")
    include("Jᴴv.jl")

    @inline global_id() = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    const THREADS_PER_BLOCK = 64

    export simulate_derivatives, Jv, Jᴴv

end
