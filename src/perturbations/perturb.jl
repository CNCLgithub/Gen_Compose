
@gen function perturb(prev_trace, rv::DynamicDistribution{T} where {T}, addr)
    choices = get_choices(prev_trace)
    value = get_value(choices, addr)
    @trace(rv(value), addr)
    return nothing
end

"""
Returns a function that will perfom an mh step over each
latent described in `moves`.
"""
function gibbs_steps(moves::AbstractArray{DynamicDistribution{T}} where {T},
                     addresses::AbstractVector)
    return trace -> foldl((t, lz, ad) -> first(mh(t, perturb, (lz, ad))),
                          zip(moves, addresses), init = trace)
end;


export gibbs_steps
