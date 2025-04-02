module MRSTAT

using BlochSimulators
using StaticArrays, LinearAlgebra, Statistics, StructArrays
using LinearMaps
using ComputationalResources
using ImagePhantoms
using PythonPlot
using QMRIColors

include("TrustRegionReflective/TrustRegionReflective.jl")
include("DerivativeOperations/DerivativeOperations.jl")

using .TrustRegionReflective
using .DerivativeOperations

include("utils/make_phantom.jl")
include("utils/objective.jl")
# include("utils/RelaxationColors.jl")
include("utils/pythonplot.jl")
include("utils/simulation_data.jl")

function mrstat_recon(
    raw_data::AbstractVector{<:SVector},
    sequence::BlochSimulator,
    coordinates,
    coil_sensitivities::AbstractVector{<:SVector},
    trajectory::CartesianTrajectory;
    x0=T₁T₂ρˣρʸ(log(1.0), log(0.100), 1.0, 0.0),
    LB=T₁T₂ρˣρʸ(log(0.1), log(0.001), -Inf, -Inf),
    UB=T₁T₂ρˣρʸ(log(7.0), log(3.000), Inf, Inf),
    trf_options=TrustRegionReflective.SolverOptions(),
)

    # Repeat the initial guess and bounds for each voxel
    # Note: x0, LB and UB are in log space for T₁ and T₂
    num_voxels = length(coordinates)
    x0 = repeat(x0', num_voxels) |> f32
    LB = repeat(LB', num_voxels) |> f32
    UB = repeat(UB', num_voxels) |> f32

    # Check coil sensitivities
    @assert all(Cᵢ -> !iszero(sum(Cᵢ)), coil_sensitivities)

    resource = CUDALibs()
    objfun = (x, mode) -> objective(x, resource, mode, raw_data, sequence, coordinates, coil_sensitivities, trajectory)

    plotfun(x, figtitle) = plot_T₁T₂ρ(optim_to_physical_pars(x), isqrt(num_voxels), isqrt(num_voxels), figtitle)
    plotfun(x0, "Initial Guess")
    # plotfun(x, figtitle) = println("Not plotting anything")

    output = TrustRegionReflective.solver(objfun, vec(x0), vec(LB), vec(UB), trf_options, plotfun)

    return output
end

function example_mrstat_recon()
    simulation_data = generate_simulation_data()
    return mrstat_recon(simulation_data...)
end

end # module MRSTAT