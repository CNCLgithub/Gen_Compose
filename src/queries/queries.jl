export Query,
    LatentMap,
    ChainDigest,
    digest

"The posterior to compute: Pr(H | O)"
abstract type Query end


const LatentMap = Dict{Symbol, Function}
# TODO: this is confusing... move or rename
const ChainDigest = Dict{Symbol, Any}

"""Extracts summaries over the current state of the inference chain"""
function digest end

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

function latents(::Query) end
function initialize_results(::Query)
    error("undefined")
end

include("static_query.jl")
include("sequential_query.jl")
# include("factorized_query.jl")
