export SequentialQuery,
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
    # A collection of observation(s) across time
    observations::Vector{Gen.ChoiceMap}
end;

initial_args(q::SequentialQuery) = q.initial_args
initial_constraints(q::SequentialQuery) = q.initial_constraints
initialize_results(q::SequentialQuery) = length(q.latents)

function Base.length(q::SequentialQuery)
    length(q.observations)
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
    # REVIEW: needed? referenced collections handle bounds
    # 1 <= i <= length(q) || throw(BoundsError(q, i))
    obs = q.observations[i]
    args = q.args[i]
    StaticQuery(q.latents, q.forward_function, args, obs)
end

latents(q::SequentialQuery) = q.latents
