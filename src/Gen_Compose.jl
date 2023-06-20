"""
Defines the types used through the inference library
"""
module Gen_Compose

using Gen
using UnPack

#################################################################################
# Chain - the estimate
##################################################################################

export InferenceChain,
    StaticChain,
    SequentialChain,
    estimand,
    estimator,
    estimate,
    auxillary,
    chain_length,
    is_finished,
    run_chain,
    run_chain!,
    resume_chain,
    report_step!


"""Data defining inference chain"""
abstract type InferenceChain end

abstract type StaticChain <: InferenceChain end

abstract type SequentialChain <: InferenceChain end

# see https://www.oxinabox.net/2020/04/19/Julia-Antipatterns.html#notimplemented-exceptions
# https://obsidian.md/
# https://docs.logseq.com/#/page/start%20here
# estimand
estimand(::InferenceChain)::InferenceQuery
# estimator
estimator(::InferenceChain)::InferenceProcedure
#estimate
estimate(::InferenceChain)

auxillary(::InferenceChain)

chain_length(::InferenceChain)::Int
chain_length(c::StaticChain) = steps(estimator(chain))
chain_length(c::SequentialChain) = length(estimand(chain))

is_finished(::InferenceChain) = chain_length(c) == i

export ChainLogger,
    buffer,
    report_step!,
    resume_chain


abstract type ChainLogger end
buffer(::chainlogger)


report_step!(::ChainLogger, ::InferenceChain, ::Int)


function run_chain(p::InferenceProcedure,
                   q::Query,
                   logger::ChainLogger = null_logger)
    # Initialized data structures that hold inference traces
    chain = initialize_chain(p, q)
    run_chain!(chain, 1, logger, path)
    return chain
end

function run_chain!(chain::InferenceChain,
                    start_idx::Int,
                    logger::ChainLogger)

    for it = start_idx:chain_length(chain)
        step!(chain, it)
        report_step!(logger, chain, it)
    end
    return nothing
end

function resume_chain(::ChainLogger)
    error("Not implemented")
end

include("loggers.jl")

#################################################################################
# Queries - the estimand
##################################################################################

"The posterior to compute: $Pr(H \mid O)$"
abstract type Query end

"The left-hand side of the conditional"
latents(::Query)


"The right-hand side of the conditional"
given(::Query)

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

include("queries/queries.jl")

#################################################################################
# Inference procedures - the estimator
#################################################################################

export InferenceProcedure,
    AuxillaryState,
    EmptyAuxState,
    steps,
    initialize_chain,
    step!,
    report_step!

"A posterior estimator"
abstract type InferenceProcedure end

"""Auxillary state for procedure"""
abstract type AuxillaryState end

"""Dummy auxillary state"""
struct EmptyAuxState <: AuxillaryState end


steps(::InferenceProcedure)::Int
function initialize_chain(::InferenceProcedure, ::Query)::InferenceChain
    error("Not implemented")
end

step!(::InferenceChain, ::Int)

include("procedures/procedures.jl")

end # module
