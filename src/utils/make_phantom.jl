# Create quantitative shepp logan phantom of size n x n
function make_phantom(N)

    sl = shepp_logan(N, SheppLoganBrainWeb()) |> rotr90

    regions = map( val -> findall(sl .== val), unique(sl))

    T₁ = zeros(N,N)
    T₂ = zeros(N,N)
    ρˣ = zeros(N,N)
    ρʸ = zeros(N,N)

    for r in regions
        T₁[r] .= rand((0.3:0.01:2.5))
        T₂[r] .= rand((0.03:0.001:0.2))
        ρˣ[r] .= rand(0.5:0.02:1.5)
        ρʸ[r] .= rand(0.5:0.02:1.5)
    end

    T₂[ T₂ .> T₁ ] .= 0.5 * T₁[ T₂ .> T₁ ]

    return map(T₁T₂ρˣρʸ, T₁, T₂, ρˣ, ρʸ) |> StructArray
end