
@gen function perturb(prev_trace, rv::LazyDistribution{T} where {T})
    choices = get_choices(prev_trace)
    value = get_value(choices, rv.addr)
    @trace(random(rv, value), rv.addr)
    return nothing
end

"""
Returns a function that will perfom an mh step over each
latent described in `moves`.
"""
function gibbs_steps(moves::AbstractArray{LazyDistribution{T}} where {T})
    return trace -> foldl((t, lz) -> first(mh(t, perturb, (lz,))),
                          moves, init = trace)
end;


export gibbs_steps
