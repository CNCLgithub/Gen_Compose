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
    obs_addr, obs_value = first(values)
    _create_seq_obs(obs_addr, obs_value)
end

"""

Defines a singule target distribution
"""
struct SequentialQuery <: Query
    # A list of selections describing the latents to infer
    latents::AbstractVector{T} where {T}
    # The forward function
    forward_function::T where T<:Gen.GenerativeFunction
    args::Vector{Tuple}
    # A numerical structure that contains the observation(s)
    observations::C where C<:Gen.ChoiceMap
    SequentialQuery(latents, forward_function, args, obs) =
        new(latents, forward_function, args, create_seq_obs(obs))
end;

initialize_results(q::SequentialQuery) = length(q.latents)
function Base.length(q::SequentialQuery)
    (_, obs) = first(Gen.get_values_shallow(q.observations))
    length(obs)
end

function observation_address(q::SequentialQuery)
    (addr, _) = first(Gen.get_values_shallow(q.observations))
    return addr
end

function Base.iterate(q::SequentialQuery, state::Int = 1)
    if state > length(q)
        return nothing
    else
        (addr, values) = first(Gen.get_values_shallow(q.observations))
        obs = choicemap()
        set_value!(obs, (addr, state), values[state])
        # nothing is the initial state of the gm
        # TODO: refine argument indexing
        args = (nothing, q.args[state]...)
        (StaticQuery(q.latents, q.forward_function, args, obs),
         state + 1)
    end
end

Base.eltype(::Type{SequentialQuery}) = StaticQuery
