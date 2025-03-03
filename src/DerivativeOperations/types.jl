abstract type ∂mˣʸ{N,T} <: FieldVector{N,T} end

struct ∂mˣʸ∂T₁T₂{T} <: ∂mˣʸ{2,T}
    ∂T₁::T
    ∂T₂::T
end

struct ∂mˣʸ∂T₁T₂ρˣρʸ{T} <: ∂mˣʸ{4,T}
    ∂T₁::T
    ∂T₂::T
    ∂ρˣ::T
    ∂ρʸ::T
end

for S in subtypes(∂mˣʸ)
    @eval StaticArrays.similar_type(::Type{$(S){T}}, ::Type{T}, s::Size{(fieldcount($(S)),)}) where T = $(S){T}
end