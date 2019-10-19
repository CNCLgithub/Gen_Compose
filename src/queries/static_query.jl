import Gen.sample

export StaticQuery,
    sample,
    initialize_results

function update_context!(c::Gen.DynamicChoiceMap , latent, new_value)
    set_value!(c, latent, new_value)
end

"""

Defines a singule target distribution
"""
struct StaticQuery{L,C,O} <: Query{L,C,O}
    # A list of selections describing the latents to infer
    latents::AbstractVector{T} where {T}
    # latents::T where T<:Gen.Selection
    # Random variables describing the prior over each latent
    prior::DeferredPrior
    # The forward function
    forward_function #::Gen.GenerativeFunction{O, U} where U<:Gen.Trace
    # A numerical structure that contains the observation(s)
    observations::O
end;

# struct SequentialQuery{L,C,O} <: Query{L,C,O}
#     # A list of selections describing the latents to infer
#     latents::AbstractVector{Gen.Selection}
#     # Random variables describing the prior over each latent
#     distributions::AbstractVector{D} where D <: LazyDistribution
#     # A choicemap containing all of the arguments need to run the forward model
#     context::C
#     # The forward function
#     forward_function::Gen.GenerativeFunction{O, U} where U<:Gen.Trace
#     # A numerical structure that contains the observation(s)
#     observations::U where U <: AbstractVector{O}
# end;

"""
    val::Context{C} = prior(q::InferenceQuery{L, C, O})

Computes P(H), drawing samples from the prior defined in `q`.
Each sample from the prior generates a `Context`,
where addresses, `Query.latents` are drawn
"Pr(h)"
"""

# @gen function prior(q::StaticQuery{L,C,O} where {L,C,O})
#     new_context = deepcopy(q.context)
#     addresses = keys(Gen.get_subselections(q.latents))
#     for (latent, dist) in zip(addresses, q.distributions)
#         println(latent)
#         new_value = @trace(dist... , latent)
#         update_context!(new_context, latent, new_value)
#     end
#     return new_context
# end


"""
    likelihood(q::InferenceQuery{L,C,O}, c::Context{C})

Computes P(O|H)

Given a context sampled from the prior over `q`, run the forward model on `c`
and compute the score of P(O,H).
"""
# @gen function likelihood(q::StaticQuery{L,C,O} where {L,C,O},
#                          context,
#                          addr::T where T<:Gen.Selection)
#     @trace(q.forward_function(context), addr)
# end


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

