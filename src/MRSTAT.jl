module MRSTAT

using BlochSimulators
using StaticArrays, LinearAlgebra, Statistics, StructArrays
using LinearMaps
using ComputationalResources
using PythonPlot

include("TrustRegionReflective/TrustRegionReflective.jl")
include("DerivativeOperations/DerivativeOperations.jl")

using .TrustRegionReflective
using .DerivativeOperations

include("utils/make_phantom.jl")
include("utils/objective.jl")
include("utils/pythonplot.jl")

function main()

    # Simulation size

    N = 224; # phantom of size N^2
    K = 5; # number of fully sampled Cartesian "transient-state k-spaces"
    nTR = K*N; # total number of TRs

    # Make sequence

    RF_train = range(start=1,stop=90,length=nTR) .|> complex;
    sliceprofiles = ones(nTR,1) .|> complex;
    TR = 0.010;
    TE = 0.006;
    max_state = 35;
    TI = 0.025;

    # assemble sequence struct
    sequence = FISP2D(RF_train, sliceprofiles, TR, TE, max_state, TI) |> f32 |> gpu;

    # Make coordinates

    FOVˣ = 22.4 # cm
    FOVʸ = 22.4 # cm
    Δx = FOVˣ/N; # cm
    Δy = FOVˣ/N; # cm
    x =  -FOVˣ/2 : Δx : FOVˣ/2 - Δx; # cm
    y =  -FOVʸ/2 : Δy : FOVʸ/2 - Δy; # cm

    coordinates = tuple.(x,y');

    # Make trajectory

    # dwell time between samples within readout
    Δt = 5e-6

    # phase encoding lines (linear sampling, repeated K times)
    py_min = -N÷2;
    py_max =  N÷2-1;
    py = repeat(py_min:py_max, K);

    # determine starting point in k-space for each readout
    Δkˣ = 2π / FOVˣ;
    Δkʸ = 2π / FOVʸ;
    k_start_readout = [(-N/2 * Δkˣ) + im * (py[r] * Δkʸ) for r in 1:nTR];

    # k-space step between samples within readout
    Δk_adc = Δkˣ

    # assemble trajectory struct
    nreadouts = nTR
    nsamplesperreadout = N

    trajectory = CartesianTrajectory(nreadouts, nsamplesperreadout, Δt, k_start_readout, Δk_adc, py)

    # Make phantom

    phantom = make_phantom(N, coordinates);

    plot_T₁T₂ρ(phantom, N, N, "Ground truth")

    # Make coil sensitivities

    ncoils = 1
    coil_sensitivities = rand((0.75:0.01:1.25), ncoils,N^2) .|> complex
    coil_sensitivities .= 1
    coil_sensitivities = map(SVector{ncoils}, eachcol(coil_sensitivities))

    # We use two different receive coils
    coil₁ = complex.(repeat(LinRange(0.8,1.2,N),1,N));
    coil₂ = coil₁';

    coil_sensitivities = map(SVector{2}, vec(coil₁), vec(coil₂))

    # Set precision and send to gpu

    phantom             = gpu(f32(vec(phantom)))
    sequence            = gpu(f32(sequence))
    trajectory          = gpu(f32(trajectory))
    coil_sensitivities  = gpu(f32(coil_sensitivities))
    coordinates         = gpu(f32(vec(coordinates)))

    # Simulate data
    resource = CUDALibs()
    raw_data = simulate_signal(resource, sequence, phantom, trajectory, coil_sensitivities)

    # Add noise?

    # Set reconstruction options

    x0 = T₁T₂ρˣρʸ(log(1.0), log(0.100),  1.0,  0.0) # note the logarithmic scaling to T1 and T2
    LB = T₁T₂ρˣρʸ(log(0.1), log(0.001), -Inf, -Inf) # note the logarithmic scaling to T1 and T2
    UB = T₁T₂ρˣρʸ(log(7.0), log(3.000),  Inf,  Inf) # note the logarithmic scaling to T1 and T2

    # Repeat x0, LB and UB for each voxel
    nr_voxels = N^2;

    x0 = repeat(x0', nr_voxels) |> f32;
    LB = repeat(LB', nr_voxels) |> f32;
    UB = repeat(UB', nr_voxels) |> f32;

    # Check that there are no points with coil sensitivity zero within the mask
    @assert all( Cᵢ -> !iszero(sum(Cᵢ)), coil_sensitivities);

    # Make plot function for further plotting of the iterations
    objfun = (x,mode) -> objective(x, resource, mode, raw_data, sequence, coordinates, coil_sensitivities, trajectory)

    # Run Trust Refion Reflective solver
    trf_min_ratio = 0.05;
    trf_max_iter = 20;
    trf_max_iter_steihaug = 20;
    trf_tol_steihaug = 0.1;
    trf_init_scale_radius = 0.1;
    trf_save_every_iter = false;

    TRF_options = TrustRegionReflective.SolverOptions(
        trf_min_ratio,
        trf_max_iter,
        trf_max_iter_steihaug,
        trf_tol_steihaug,
        trf_init_scale_radius,
        trf_save_every_iter)

    plotfun(x, figtitle) = plot_T₁T₂ρ(optim_to_physical_pars(x), N, N, figtitle)

    plotfun(x0, "Initial Guess")

    # Run non-linear solver

    output = TrustRegionReflective.solver(objfun, vec(x0), vec(LB), vec(UB), TRF_options, plotfun)

    return output
end


end
