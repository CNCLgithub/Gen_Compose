
"Describes the left hand of the conditional"
struct Context{D}
    "The variables that define the scene"
    data::D
    "The arguments to pass to the generative model in addition to `data`"
    args<:Tuple{Any}
end;

function update_context!(c<:Gen.ChoiceMap , latent, new_value)
    set_value!(c, latent, new_value)
end

struct Query{L,C,O} <: InferenceQuery{L,C,O}
    # A list of selections describing the latents to infer
    latents<:AbstractVector{Gen.Selection}
    # Random variables describing the prior over each latent
    distributions <: AbstractVector{D} where D <: Distribution{L}
    # A choicemap containing all of the arguments need to run the forward model
    context::C
    # The forward function
    forward_function<:Gen.GenerativeFunction{O, U <: Gen.Trace}
    # A numerical structure that contains the observation(s)
    observations::O
end;

struct SequentialQuery{L,C,O} <: Query{L,C,O} \
    where {O <: AbstraceVector{T}}
end

# function prior(q::InferenceQuery{L,C,O}) => C
#     ...
# end

"""
Each sample from the prior generates a `Context`,
where addresses, `Query.latents` are drawn
"Pr(h)"
"""
@gen function prior(q::Query{C,O,L}) => C
    new_context = copy(q.context)
    for (latent, dist) in zip(q.latents, q.distributions)
        new_value = @trace(dist, (,), latent)
        update_context!(new_context, latent, new_value)
    end
    return new_context
end


@gen function likelihood(q::Query{C, O}, context)
    @trace(q.forward_function(context), :obs)
end


@gen function sample(q::Query{C, O})
    c = prior(q)
    likelihood(q, c)
end
