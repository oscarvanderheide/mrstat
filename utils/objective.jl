function objective(optimpars::Vector{<:Real}, resource, mode, raw_data, sequence, coordinates, coil_sensitivities, trajectory)

    # We compute the residual rᵢ = ||d Σᵢ (dᵢ - M(T₁,T₂,B₁,B₀)*Cᵢ*ρ)
    # f = (1/2) * |r|^2
    # The gradient is computed as g = ℜ(Jᴴr)

    # mode 0 -> compute f and r only
    # mode 1 -> compute f, r and g
    # mode 2 -> compute f, r, g and assemble approximate Hessian

    # Convert optimpars (Vector{<:Real}) to Vector{<:AbstractTissueParameters} to be used in simulations
    parameters = optim_to_physical_pars(optimpars, coordinates)

    # Convert to single precision and send to gpu device
    parameters = gpu(f32(parameters))

    # Compute magnetization at echo times
    echos = simulate_magnetization(resource, sequence, parameters)

    if mode == 0

        # Apply phase encoding
        phase_encoding!(echos, trajectory, parameters)

        # Compute signal
        s = magnetization_to_signal(resource, echos, parameters, trajectory, coil_sensitivities)

        # Compute residual r
        r = s - raw_data;

        # Compute cost f
        f = 0.5 * sum(norm.(r).^2);

        return f, r

    elseif mode > 0

        # Compute partial derivatives of magnetization at echo time
        ∂echos = simulate_derivatives(echos, resource, sequence, parameters)

        # Apply phase encoding
        phase_encoding!(echos, trajectory, parameters)
        phase_encoding!(∂echos, trajectory, parameters)

        # Compute signal
        s = magnetization_to_signal(resource, echos, parameters, trajectory, coil_sensitivities)

        # Compute residual r
        r = s - raw_data;

        # Compute cost f
        f = 0.5 * sum(norm.(r).^2);

        # Compute gradient
        g = Jᴴv(resource, echos, ∂echos, parameters, coil_sensitivities, trajectory, r)
        g = StructArray(g)
        g = reduce(vcat,fieldarrays(g))
        g = real.(g)
        g = collect(g)

        mode == 1 && return f, r, g

        # Make Gauss-Newton matrix multiply function
        reJᴴJ(x) = begin
            np = 4 # nr of reconstruction parameters per voxel
            x = reshape(x,:,np)
            x = map(SVector{4}, eachcol(x)...) |> gpu
            y = Jv(resource, echos, ∂echos, parameters, coil_sensitivities, trajectory, x)
            z = Jᴴv(resource, echos, ∂echos, parameters, coil_sensitivities, trajectory, y)
            return real.(reduce(vcat,fieldarrays(StructArray(z))))
        end

        H = LinearMap(
            v -> reJᴴJ(v),
            v -> v, # adjoint operation not used
        length(g),length(g));

        return f, r, g, H
    end
end

function optim_to_physical_pars(optimpars)

    optimpars = reshape(optimpars,:,4)
    T₁ = exp.(optimpars[:,1])
    T₂ = exp.(optimpars[:,2])
    ρˣ = optimpars[:,3]
    ρʸ = optimpars[:,4]

    return map(T₁T₂ρˣρʸ, T₁, T₂, ρˣ, ρʸ)
end

function optim_to_physical_pars(optimpars, coordinates)

    optimpars = reshape(optimpars,:,4)
    T₁ = exp.(optimpars[:,1])
    T₂ = exp.(optimpars[:,2])
    ρˣ = optimpars[:,3]
    ρʸ = optimpars[:,4]
    x = first.(collect(coordinates))
    y = last.(collect(coordinates))

    return map(T₁T₂ρˣρʸxy, T₁, T₂, ρˣ, ρʸ, x, y)
end