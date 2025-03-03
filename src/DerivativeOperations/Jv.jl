function Jv(resource::CUDALibs,
    echos,
    ∂echos,
    parameters,
    coil_sensitivities::AbstractArray{SVector{Nc}{T}},
    trajectory,
    v) where {Nc,T<:Complex}

    # assumes all structs/arrays are sent to GPU before

    # allocate output on GPU
    Jv = CUDA.zeros(eltype(coil_sensitivities), nsamples(trajectory))
    # launch cuda kernel
    nr_blocks = cld(nsamples(trajectory), THREADS_PER_BLOCK)

    CUDA.@sync begin
        @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK Jv_kernel!(Jv, echos, ∂echos, parameters, coil_sensitivities, trajectory, v)
    end

    return Jv
end

function Jv_kernel!(Jv, echos, ∂echos, parameters, coil_sensitivities::AbstractArray{SVector{Nc}{T}}, trajectory,v) where {T,Nc}

    i = global_id() # global sample point index

    # sequence constants
    ns = trajectory.nsamplesperreadout # nr of samples per readout
    nr = trajectory.nreadouts # nr of readouts

    if i <= nr * ns

        r = fld1(i,ns) # determine which readout
        s = mod1(i,ns) # determine which sample within readout
        nv = length(parameters) # nr of voxels
        # v is assumed to be an array of ... SVectors?
        jv = zero(MVector{Nc}{T})

        @inbounds for voxel ∈ 1:nv

            # load coordinates, parameters, coilsensitivities and proton density for voxel
            p = parameters[voxel]
            ρ = complex(p.ρˣ,p.ρʸ)
            # x,y = coordinates[voxel]
            C = coil_sensitivities[voxel]
            # R₂ = inv(p.T₂)
            # load magnetization and partial derivatives at echo time of the r-th readout
            m  =  echos[r,voxel]
            ∂m = ∂echos[r,voxel]
            # compute decay (T₂) and rotation (gradients and B₀) to go to sample point
            m, ∂m = ∂to_sample_point(m, ∂m, trajectory, r, s, p)
            # store magnetization from this voxel, scaled with v (~ proton density) and C in accumulator
            ∂mv = v[voxel] .* ∂mˣʸ∂T₁T₂ρˣρʸ(∂m.∂T₁, ∂m.∂T₂, m, m*im)
            for c in eachindex(C)
                lin_scale = SVector{4}(p.T₁*C[c]*ρ, p.T₂*C[c]*ρ, C[c], C[c])
                jv[c] += sum(lin_scale .* ∂mv)
            end

        end # loop over voxels

        @inbounds Jv[i] = SVector(jv)
    end

    nothing
end

@inline function ∂to_sample_point(mₑ, ∂mₑ::∂mˣʸ∂T₁T₂, trajectory::CartesianTrajectory, readout_idx, sample_idx, p)

    # Read in constants
    R₂ = inv(p.T₂)
    ns = nsamplesperreadout(trajectory, readout_idx)
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    x = p.x

    # There are ns samples per readout, echo time is assumed to occur
    # at index (ns÷2)+1. Now compute sample index relative to the echo time
    s = sample_idx - ((ns÷2)+1)
    # Apply readout gradient, T₂ decay and B₀ rotation
    θ = Δkₓ * x
    hasB₀(p) && (θ += π*p.B₀*Δt*2)
    E₂eⁱᶿ = exp(-s*Δt*R₂ + im*s*θ)

    ∂E₂eⁱᶿ = ∂mˣʸ∂T₁T₂(0, (s*Δt)*R₂*R₂*E₂eⁱᶿ)

    ∂mₛ = ∂mₑ * E₂eⁱᶿ + mₑ * ∂E₂eⁱᶿ
    mₛ = E₂eⁱᶿ * mₑ

    return mₛ,∂mₛ
end