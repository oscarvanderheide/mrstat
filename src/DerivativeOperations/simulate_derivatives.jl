# Hard-coded to simulate T₁ and T₂ derivatives using finite differences.
function simulate_derivatives(m, resource, sequence, parameters, Δ = 10^-4 )

    Δ = Float32(Δ)
    # parameters = collect(parameters)
    base_T₁ = collect(parameters.T₁)
    base_T₂ = collect(parameters.T₂)

    # derivatives w.r.t. T₁
    Δpars = map((T₁,T₂) -> T₁T₂(T₁+Δ, T₂), base_T₁, base_T₂) |> gpu
    Δm = simulate_magnetization(resource, sequence, Δpars)
    ∂m∂T₁ = @. (Δm - m)/Δ

    # derivatives w.r.t. T₂
    # Δpars = map(p -> T₁T₂(p.T₁, p.T₂+Δ), parameters) |> gpu
    Δpars = map((T₁,T₂) -> T₁T₂(T₁, T₂+Δ), base_T₁, base_T₂) |> gpu
    Δm = simulate_magnetization(resource, sequence, Δpars)
    ∂m∂T₂ = @. (Δm - m)/Δ

    ∂m = map((∂T₁,∂T₂)-> ∂mˣʸ∂T₁T₂(∂T₁,∂T₂), ∂m∂T₁, ∂m∂T₂)

    return ∂m
end