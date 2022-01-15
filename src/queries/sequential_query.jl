export SequentialQuery,
    latents,
    initialize_results

"""

Defines a singule target distribution
"""
struct SequentialQuery <: Query
    # A list of selections describing the latents to infer
    latents::LatentMap
    # The forward function
    forward_function::T where T<:Gen.GenerativeFunction
    initial_args::Tuple
    initial_constraints::Gen.ChoiceMap
    args::Vector{Tuple}
    # A numerical structure that contains the observation(s)
    observations::Vector{Gen.ChoiceMap}
end;

latents(q::SequentialQuery) = q.latents
initialize_results(q::SequentialQuery) = length(q.latents)
initial_args(q::SequentialQuery) = q.initial_args
initial_constraints(q::SequentialQuery) = q.initial_constraints

# -- Implementing Base.Iterable -- #

function Base.length(q::SequentialQuery)
    min(length(q.observations), length(q.args))
end

function Base.iterate(q::SequentialQuery, state::Int = 1)
    if state > length(q)
        return nothing
    else
        # TODO: refine argument indexing
        (q[state], state + 1)
    end
end

Base.eltype(::Type{SequentialQuery}) = StaticQuery

function Base.getindex(q::SequentialQuery, i::Int)
    1 <= i <= length(q) || throw(BoundsError(q, i))
    obs = q.observations[i]
    args = q.args[i]
    StaticQuery(q.latents, q.forward_function, args, obs)
end

