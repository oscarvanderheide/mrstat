
function Jᴴv(::CUDALibs, echos::AbstractArray{T}, ∂echos, parameters, coil_sensitivities, trajectory, coordinates, v) where {T<:Complex}

    # assumes all structs/arrays are sent to GPU before

    # allocate output on GPU
    nv = length(parameters)

    Jᴴv = CUDA.zeros(∂mˣʸ∂T₁T₂ρˣρʸ{T}, nv)

    # launch cuda kernel
    nr_blocks = cld(nv, THREADS_PER_BLOCK)

    CUDA.@sync begin
        @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK Jᴴv_kernel!(
            Jᴴv, 
            echos, 
            ∂echos,
            parameters, 
            coil_sensitivities, 
            trajectory, 
            coordinates, 
            v
        )
    end

    return Array(Jᴴv)
end

function test_kernel()

    return nothing
end

function Jᴴv_kernel!(Jᴴv, echos::AbstractArray{T}, ∂echos, parameters, coil_sensitivities, trajectory, coordinates, v) where {T<:Complex}

    # v is a vector of length "nr of measured samples",
    # and each element is a StaticVector of length "nr of coils"
    # output is a vector of length "nr of voxels" with each element
    
    voxel = global_id()

    @inbounds if voxel <= length(parameters)

        # sequence constants
        nr = nreadouts(trajectory) # nr of readouts

        # load parameters and spatial coordinates
        p = parameters[voxel]
        crd = coordinates[voxel]
        c = coil_sensitivities[voxel]

        # accumulators
        mᴴv = zero(SMatrix{1,NUM_COILS}{T}{NUM_COILS})
        ∂mᴴv = zero(SMatrix{2,NUM_COILS}{T}{2*NUM_COILS})

        t = 1

        for readout = 1:nr

            # load magnetization and partial derivatives at echo time of the r-th readout
            mₑ = echos[readout,voxel]
            ∂mₑ = ∂echos[readout,voxel]

            mᴴv, ∂mᴴv, t = ∂expand_readout_and_accumulate_mᴴv(mᴴv, ∂mᴴv, mₑ, ∂mₑ, p, crd.x, trajectory, readout, t, v)

        end # loop over readouts

        tmp = vcat(∂mᴴv, mᴴv, -im*mᴴv) # size = (nr_nonlinpars + 2) x nr_coils
        Jᴴv[voxel] = zero(eltype(Jᴴv))

        ρ = complex(p.ρˣ,p.ρʸ)

        for j in eachindex(c)
            lin_scale = SVector{4}(p.T₁*c[j]*ρ, p.T₂*c[j]*ρ, c[j], c[j])
            Jᴴv[voxel] += conj(lin_scale) .* tmp[:,j]
        end
    end

    # At this point, output is a vector of structs/svectors of the form (∂T₁,∂T₂,...,∂ρˣ,∂ρʸ)
    # Convert to a StructArray partial derivatives w.r.t. T₁ can be accessed with .T₁, etc.
    # output = StructArray(output)

    return nothing
end

@inline function ∂expand_readout_and_accumulate_mᴴv(mᴴv, ∂mᴴv, mₑ, ∂mₑ, p, x,trajectory::CartesianTrajectory2D, readout, t, v)

    ns = trajectory.nsamplesperreadout
    Δt = trajectory.Δt
    Δk = trajectory.Δk_adc
    R₂ = inv(p.T₂)

    # Gradient rotation per sample point
    θ = Δk * x
    # B₀ rotation per sample point
    θ += (hasB₀(p) ? Δt*π*p.B₀*2 : 0)
    # "Rewind" to start of readout
    R = exp((ns÷2)*Δt*R₂ - im*(ns÷2)*θ)
    ∂R = ∂mˣʸ∂T₁T₂(0, -(ns÷2)*Δt*R₂*R₂*R)

    ∂mₛ = ∂mₑ * R + mₑ * ∂R
    mₛ = mₑ * R
    # T₂ decay and gradient- and B₀ induced rotation per sample
    E₂eⁱᶿ = exp(-Δt*R₂ + im*θ)
    ∂E₂eⁱᶿ = ∂mˣʸ∂T₁T₂(0, Δt*R₂*R₂*E₂eⁱᶿ)

    for sample in 1:ns
        # accumulate dot product in mᴴv
        mᴴv  += conj(mₛ)   * transpose(v[t])
        ∂mᴴv += conj(∂mₛ) .* transpose(v[t])
        # compute magnetization at next sample point
        ∂mₛ = ∂mₛ * E₂eⁱᶿ + mₛ * ∂E₂eⁱᶿ
        mₛ  = mₛ * E₂eⁱᶿ
        # increase time index
        t += 1
    end

    return mᴴv, ∂mᴴv, t
end