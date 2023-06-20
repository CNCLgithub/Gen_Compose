export MCMC, MCMCTrace, MetropolisHastings, MHTrace, StaticMHChain

abstract type MCMC <: InferenceProcedure end

"""
Static MH chain
"""
mutable struct StaticMHChain <: StaticChain
    query::StaticQuery
    proc::MCMC
    state::Gen.Trace
    auxillary::AuxillaryState
end

estimand(c::StaticMHChain) = c.query
estimator(c::StaticMHChain) = c.proc
estiamte(c::StaticMHChain) = c.state

"""
Simple definition of MH procedure
"""
struct MetropolisHastings <: MCMC
    samples::Int
    update::T where T<:Function
end

steps(p::MetropolisHastings) = p.samples

isfinished(p::MetropolisHastings, i::Int) = i == steps(p)

function initialize_procedure(proc::MCMC,
                              query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    trace
end

function initialize_chain(proc::MCMC,
                          query::StaticQuery)
    trace = initialize_procedure(proc, query)
    return StaticMHChain(query, proc, trace, EmptyAuxState())
end

function step!(chain::StaticMHChain,
               idx::Int)
    step!(chain, estimator(chain), idx)
end
function step!(chain::StaticMHChain,
               proc::MetropolisHastings,
               idx::Int)
    @unpack proc, state = chain
    chain.state = proc.update(state)
    return nothing
end
