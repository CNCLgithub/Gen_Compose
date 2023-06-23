export StaticQuery,
    SequentialQuery

"""

Defines a single target distribution
"""
struct StaticQuery <: Query
    # A collection of latents in the left hand of the posterior
    latents::LatentMap
    # The forward function
    forward_function::T where T<:Gen.GenerativeFunction
    args::T where T<:Tuple
    # A numerical structure that contains the observation(s)
    observations::Gen.ChoiceMap
end

latents(q::StaticQuery) = q.latents
given(q::StaticQuery) = (q.forward_function, q.args, q.observations)

@doc raw"""
Defines a conditional where ``Pr(H \mid O)`` is factorized into a series of:
```math

Pr(H_0) * \prod^T_i Pr(H_i \mid O_i) Pr(H_i \mid H_{i-1})
```
"""
struct SequentialQuery <: Query
    # A list of selections describing the latents to infer
    latents::LatentMap
    # The forward function
    forward_function::T where T<:Gen.GenerativeFunction
    initial_args::Tuple
    initial_constraints::Gen.ChoiceMap
    args::Vector{Tuple}
    argdiffs::Vector{Tuple}
    # A collection of observation(s) across time
    observations::Vector{Gen.ChoiceMap}
end

"""
    SequentialQuery(latents, forward_function, initial_args,
                    initial_constraints, args, obs)

Construct a `SequentialQuery` with defaul argdiffs (all unknown change).
"""
function SequentialQuery(latents, ff, iargs, ics, args, obs)
    nargs = length(args)
    nobs = length(obs)
    argdiffs = fill((fill(UnknownChange(), nargs)), nobs)
end

latents(q::SequentialQuery) = q.latents
initial_args(q::SequentialQuery) = q.initial_args
initial_constraints(q::SequentialQuery) = q.initial_constraints

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

