export SequentialQuery,
    initialize_results,
    observation_address

function _create_seq_obs(addr, obs::T where T<:AbstractVector)
    c = Gen.choicemap()
    set_value!(c, addr, obs)
    Gen.StaticChoiceMap(c)
end

function create_seq_obs(c::T where T<:Gen.ChoiceMap)
    values = Gen.get_values_shallow(c)
    if length(values) != 1
        error("Observation must have one shallow address")
    end
    _create_seq_obs(first(values)...)
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
    SequentialQuery(latents, prior, forward_function, args, obs) =
        new(latents, prior, forward_function, args, create_seq_obs(obs))
end;

function gen_target_query(q, t, obs, addr)
    c = choicemap()
    set_value!(c, addr, obs)
    args = (nothing, t, q.args...)
    return  StaticQuery(q.latents,
                        q.prior,
                        q.forward_function,
                        args,
                        c)
end

function atomize(q::SequentialQuery)
    (addr, values) = first(Gen.get_values_shallow(q.observations))
    map(tx -> gen_target_query(q, tx..., addr),
        enumerate(values))
end

initialize_results(q::SequentialQuery) = length(q.latents)
function Base.length(q::SequentialQuery)
    (_, obs) = first(Gen.get_values_shallow(q.observations))
    length(obs)
end

function observation_address(q::SequentialQuery)
    (addr, _) = first(Gen.get_values_shallow(q.observations))
    return addr
end
