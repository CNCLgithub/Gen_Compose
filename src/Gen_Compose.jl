"""
Defines the types used through the inference library
"""
module Gen_Compose

using Gen
using UnPack

# see https://www.oxinabox.net/2020/04/19/Julia-Antipatterns.html#notimplemented-exceptions
#################################################################################
# Exports
##################################################################################

export InferenceChain,
    Query,
    LatentMap,
    ChainDigest,
    InferenceProcedure,
    AuxillaryState,
    ChainLogger,
    estimand,
    estimator,
    estimate,
    auxillary,
    step,
    steps,
    is_finished,
    run_chain,
    run_chain!,
    step!,
    resume_chain,
    report_step!,
    buffer,
    report_step!,
    resume_chain,
    digest

#################################################################################
# Types
##################################################################################

@doc raw"The estimand to compute: ``Pr(H \mid O)``"
abstract type Query end

"An estimator"
abstract type InferenceProcedure end

"An application of estimator to the estimand"
abstract type InferenceChain{Query, InferenceProcedure} end

"A mapping to digest components of an query during inference"
const LatentMap = Dict{Symbol, Function}

"A container for digested components"
const ChainDigest = Dict{Symbol, Any}

"""Auxillary state for procedure"""
abstract type AuxillaryState end

"A utility to process chain states"
abstract type ChainLogger end

#################################################################################
# Chain - the estimate
##################################################################################

"""
    estimand(chain)

Denotes the target of inference.
"""
function estimand end

"""
    estimator(chain)

The approximator over the estimand.
"""
function estimator end

"""
    estimate(chain)

The result of applying the estimator to the estimand.
"""
function estimate end


"""
    auxillary(chain)

Additional state of the chain other than the estimate.
"""
function auxillary end

"""
    step(chain)::Int

The current iteration of the chain.
"""
function step end

"""
    is_finished(chain)::Bool

Whether the chain is finished.
"""
is_finished(c::InferenceChain) = steps(c) == step(c)

"""
    initialize_chain(::InferenceProcedure, ::Query)::InferenceChain

Initializes an inference chain for the procedure, query context.
"""
function initialize_chain end

"""
    step!(::InferenceChain)

Advances the chain to the next outer step.
"""
function step! end



"""
    buffer(::ChainLogger)

Returns the buffer that stores intermediate states of the inference chain
"""
function buffer end


"""
    report_step!(::ChainLogger, ::InferenceChain, ::Int)

Makes a recor of the chain at the current step.
"""
function report_step! end


"""
    run_chain(proc, query, n, [logger])

Initializes and runs an inference chain, applying the inference
procedure to the query for `n` steps.

Optionally, records intermediate chain states via `logger`.
"""
function run_chain(p::InferenceProcedure,
                   q::Query,
                   n::Int,
                   logger::ChainLogger = NullLogger())
    # Initialized data structures that hold inference traces
    chain = initialize_chain(p, q, n)
    run_chain!(chain, logger)
    return chain
end

function run_chain!(chain::InferenceChain,
                    logger::ChainLogger)

    while !is_finished(chain)
        step!(chain)
        report_step!(logger, chain)
    end
    return nothing
end

"""
    resume_chain(::ChainLogger)::InferenceChain

Resumes and completes inference from a logger checkpoint.
"""
function resume_chain end

include("loggers.jl")

#################################################################################
# Queries - the estimand
##################################################################################

"""
    latents(::Query)

The left-hand side of the conditional
"""
function latents end

"""
    given(::Query)

The right-hand side of the conditional
"""
function given end


"""
    digest(lm, chain)

Applies a latent map to a chain.
"""
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

include("queries.jl")

#################################################################################
# Inference procedures - the estimator
#################################################################################

include("procedures/procedures.jl")

end # module
