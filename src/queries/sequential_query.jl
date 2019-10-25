import Gen.sample

export SequentialQuery,
    initialize_results,
    observation_address

function create_obs_choicemap(c::T{D} where {T<:Gen.ChoiceMap, D<:AbstractVector})
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
struct SequentialQuery <: Query
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

function gen_target_query(obs, addr)
    c = choicemap()
    set_value!(c, addr, obs)
    Gen.StaticChoiceMap(c)
end

function atomize(q::SequentialQuery)
    (addr, values) = first(Gen.get_values_shallow(q.observations))
    target_queries = Vector{Gen.StaticChoiceMap}(undef, length(values))
    map(x -> gen_target_query(x, addr), values)
end

initialize_results(q::SequentialQuery) = length(q.latents)

function observation_address(q::SequentialQuery)
    (addr, _) = first(Gen.get_values_shallow(q.observations))
    return addr
end
