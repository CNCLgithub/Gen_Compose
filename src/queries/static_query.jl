export StaticQuery,
    initialize_results,
    observation_address

function create_obs_choicemap(c::T where T<:Gen.ChoiceMap)
    # sc = Gen.StaticChoiceMap(c)
    values = Gen.get_values_shallow(c)
    if length(values) != 1
        error("Observation must have one shallow address")
    end
    return c
end

"""

Defines a singule target distribution
"""
struct StaticQuery <: Query
    # A collection of latents in the left hand of the posterior
    latents::LatentMap
    # The forward function
    forward_function::T where T<:Gen.GenerativeFunction
    args::T where T<:Tuple
    # A numerical structure that contains the observation(s)
    observations::Gen.ChoiceMap
    StaticQuery(selection, gm, args, obs) = new(selection, gm, args, obs)
                                                 # create_obs_choicemap(obs))
end

initialize_results(q::StaticQuery) = length(q.latents)

latents(q::StaticQuery) = q.latents
# function observation_address(q::StaticQuery)
#     (addr, _) = first(Gen.get_values_shallow(q.observations))
#     return addr
# end



# """
#    sample(q::InferenceQuery{L,C,O})

# Computes P(H|O)

# Samples a `c::Context{C}` from the prior and scores the likelihood.
# """
# @gen function sample(q::StaticQuery,
#                      addr::Symbol)
#     @trace(q.forward_function(q.prior), addr)
#     return nothing
# end
