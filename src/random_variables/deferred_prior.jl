export DeferredPrior,
    draw

struct DeferredPrior
    latents::AbstractVector{T} where {T}
    distributions::AbstractVector{D} where D <: StaticDistribution
end

"""
Helper that ensures type safety
"""
function extract_rv(p::DeferredPrior, addr)
    idx = findfirst(x -> x == addr, p.latents)
    p.distributions[idx]
end
@gen function draw(p::DeferredPrior, addr)
    rv = extract_rv(p, addr)
    val = @trace(rv(), addr)
    return val
end

