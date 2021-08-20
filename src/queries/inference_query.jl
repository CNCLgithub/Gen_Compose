export Query,
    LatentMap,
    ChainDigest,
    digest

"The posterior to compute: Pr(H | O)"
abstract type Query end


# """
#     val::Context{C} = prior(q::InferenceQuery{L, C, O})

# Computes P(H), drawing samples from the prior defined in `q`.
# """
# function prior end

# """
#     likelihood(q::InferenceQuery{L,C,O}, c::Context{C})

# Computes P(O|H)

# Given a context sampled from the prior over `q`, run the forward model on `c`
# and compute the score of P(O,H).
# """
# function likelihood end


# """
#    sample(q::InferenceQuery{L,C,O})

# Computes P(H|O)

# Samples a `c::Context{C}` from the prior and scores the likelihood.
# """
# function sample end

const LatentMap = Dict{Symbol, Function}
const ChainDigest = Dict{Symbol, Any}

function digest(lm::LatentMap, chain::InferenceChain)
    d = ChainDigest()
    for (k,f) in lm
        d[k] = f(chain)
    end
    return d
end

function digest(q::Query, chain::InferenceChain)
    lm = latents(q)
    digest(lm, chain)
end
# export prior
# export likelihood
# export sample

include("static_query.jl")
include("sequential_query.jl")
# include("factorized_query.jl")
