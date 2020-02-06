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
    latents::Dict{Symbol, Function}
    # The forward function
    forward_function::T where T<:Gen.GenerativeFunction
    initial_args::Tuple
    initial_constraints::Gen.ChoiceMap
    args::Vector{Tuple}
    # A numerical structure that contains the observation(s)
    observations::Vector{Gen.ChoiceMap}
    # SequentialQuery(latents, forward_function, args, obs) =
    #     new(latents, forward_function, args, create_seq_obs(obs))
end;

initial_args(q::SequentialQuery) = q.initial_args
initial_constraints(q::SequentialQuery) = q.initial_constraints

initialize_results(q::SequentialQuery) = length(q.latents)
function Base.length(q::SequentialQuery)
    min(length(q.observations), length(q.args))
end

function observation_address(q::SequentialQuery)
    (addr, _) = first(Gen.get_values_shallow(q.observations))
    return addr
end

function Base.iterate(q::SequentialQuery, state::Int = 1)
    if state > length(q)
        return nothing
    else
        # TODO: refine argument indexing
        obs = q.observations[state]
        args = q.args[state]
        latents = [m(state) for (_,m) in q.latents]
        (StaticQuery(latents, q.forward_function, args, obs),
         state + 1)
    end
end

Base.eltype(::Type{SequentialQuery}) = StaticQuery
