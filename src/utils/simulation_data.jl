function generate_simulation_data(; N=224, K=5)
    # total number of TRs
    nTR = K * N

    # Make sequence
    RF_train = range(start=1, stop=90, length=nTR) .|> complex
    sliceprofiles = ones(nTR, 1) .|> complex
    TR = 0.010
    TE = 0.006
    max_state = 32
    TI = 0.025

    sequence = FISP2D(RF_train, sliceprofiles, TR, TE, max_state, TI) |> f32 |> gpu

    # Make coordinates
    FOVˣ = 22.4 # cm
    FOVʸ = 22.4 # cm
    Δx = FOVˣ / N # cm
    Δy = FOVˣ / N # cm
    x = -FOVˣ/2:Δx:FOVˣ/2-Δx |> collect # cm
    y = -FOVʸ/2:Δy:FOVʸ/2-Δy |> collect # cm
    z = [0.0]
    coordinates = @coordinates x y z # Takes Iterators.product and convert to a StructArray of BlochSimulators.Coordinates

    # Make trajectory
    Δt = 5e-6
    py_min = -N ÷ 2
    py_max = N ÷ 2 - 1
    py = repeat(py_min:py_max, K)

    Δkˣ = 2π / FOVˣ
    Δkʸ = 2π / FOVʸ
    k_start_readout = [(-N / 2 * Δkˣ) + im * (py[r] * Δkʸ) for r in 1:nTR]
    Δk_adc = Δkˣ

    trajectory = CartesianTrajectory2D(nTR, N, Δt, k_start_readout, Δk_adc, py, 1)

    # Make phantom
    phantom = make_phantom(N)

    # Make coil sensitivities
    coil₁ = complex.(repeat(LinRange(0.8, 1.2, N), 1, N))
    if NUM_COILS == 1
        coil_sensitivities = reshape(coil₁, N^2, 1)
    elseif NUM_COILS == 2
        coil₂ = coil₁'
        coil_sensitivities = [vec(coil₁) ;; vec(coil₂)]
    else
        @error "NUM_COILS is not 1 or 2"
    end

    # Set precision and send to gpu
    phantom = gpu(f32(vec(phantom)))
    sequence = gpu(f32(sequence))
    trajectory = gpu(f32(trajectory))
    coil_sensitivities = gpu(f32(coil_sensitivities))
    coordinates = gpu(f32(vec(coordinates)))

    # Simulate data
    resource = CUDALibs()

    raw_data = simulate_signal(resource, sequence, phantom, trajectory, coordinates, coil_sensitivities)

    return raw_data, sequence, coordinates, coil_sensitivities, trajectory
end

