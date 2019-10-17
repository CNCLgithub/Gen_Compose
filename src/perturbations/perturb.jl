
@gen function perturb(prev_trace, rv::LazyDistribution{T} where {T}, addr)
    choices = get_choices(prev_trace)
    value = get_value(choices, addr)
    @trace(random(rv, value), addr)
    return nothing
end

"""
Returns a function that will perfom an mh step over each
latent described in `moves`.
"""
function gibbs_steps(moves::AbstractArray{LazyDistribution{T}} where {T},
                     addresses::G where G<:AbstractVector{Any})
    return trace -> foldl((t, lz, ad) -> first(mh(t, perturb, (lz, ad))),
                          zip(moves, addresses), init = trace)
end;


export gibbs_steps
