module TrustRegionReflective

    # Registered modules
    using Memento
    using LinearAlgebra
    using StaticArrays
    using Statistics
    using TickTock

    struct SolverOptions{T,I,B}
        min_ratio::T
        max_iter_trf::I
        max_iter_steihaug::I
        tol_steihaug::T
        init_scale_radius::T
        save_every_iter::B
    end

    mutable struct SolverOutput
        x # parameters
        f # cost
        r # residual vector
        t # elapsed time (s)
    end

    include("utils.jl")
    include("solver.jl")
    include("steihaug.jl")

end # module
