import Gen.sample

export StaticQuery,
    sample,
    initialize_results,
    observation_address

function create_obs_choicemap(c::T where T<:Gen.ChoiceMap)
    sc = Gen.StaticChoiceMap(c)
    values = Gen.get_values_shallow(sc)
    if length(values) != 1
        error("Observation must have one shallow address")
    end
    return sc
end

"""

Defines a singule target distribution
"""
struct StaticQuery <: Query
    # A list of selections describing the latents to infer
    latents::AbstractVector{T} where {T}
    # latents::T where T<:Gen.Selection
    # Random variables describing the prior over each latent
    prior::DeferredPrior
    # The forward function
    forward_function::T where T<:Gen.GenerativeFunction
    args::T where T<:Tuple
    # A numerical structure that contains the observation(s)
    observations::C where C<:Gen.ChoiceMap
    StaticQuery(latents, prior, forward_function, args, obs) =
        new(latents, prior, forward_function, args, create_obs_choicemap(obs))
end;


"""
   sample(q::InferenceQuery{L,C,O})

Computes P(H|O)

Samples a `c::Context{C}` from the prior and scores the likelihood.
"""
@gen function sample(q::StaticQuery,
                     addr::Symbol)
    @trace(q.forward_function(q.prior), addr)
    return nothing
end

initialize_results(q::StaticQuery) = length(q.latents)

function observation_address(q::StaticQuery)
    (addr, _) = first(Gen.get_values_shallow(q.observations))
    return addr
end
