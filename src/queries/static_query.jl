export StaticQuery,
    initialize_results,
    observation_address

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
end

initialize_results(q::StaticQuery) = length(q.latents)
latents(q::StaticQuery) = q.latents
