
@gen function perturb(prev_trace, rv::DynamicDistribution, addr)
    choices = get_choices(prev_trace)
    value = get_value(choices, addr)
    @trace(rv(value), addr)
    return nothing
end
"""
Returns a function that folds a trace over a collection moves given
a trace and parameters for those perturbation functions
"""
function mh_rejuvenate(moves::Vector{T} where T<:Gen.GenerativeFunction)
    return (trace, params) -> foldl((t, etc) -> first(mh(t, etc...)),
                                     zip(moves, params), init = trace)
end;


"""
Perturb each latent sequentially
"""
function gibbs_steps(rvs::Vector{DynamicDistribution{T}} where {T},
                     addresses::Vector)
    n_latents = length(rvs)
    blocks = mh_rejuvenate(repeat([perturb], n_latents))
    return trace -> blocks(trace, zip(rvs, addresses))
end;


export gibbs_steps
