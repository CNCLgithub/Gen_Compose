export MCMC, MCMCTrace, MetropolisHastings, MHTrace, StaticMHChain

abstract type MCMC <: InferenceProcedure end

"""
Simple definition of MH procedure
"""
struct MetropolisHastings <: MCMC
    update::T where T<:Function
end

"""
Static MH chain
"""
mutable struct MHChain{Q} <: InferenceChain{Q, MCMC}
    query::Q
    proc::MCMC
    state::Gen.Trace
    auxillary::AuxillaryState
    step::Int
    steps::Int

    function MHChain{Q}(q::Q,
                        p::MCMC,
                        n::Int,
                        i::Int = 1) where {Q<:Query}
        state = initialize_procedure(p, q)
        aux = EmptyAuxState()
        return new(q, p, state, aux, i, n)
    end
end

estimand(c::MHChain) = c.query
estimator(c::MHChain) = c.proc
estimate(c::MHChain) = c.state
auxillary(c::MHChain) = c.auxillary
step(c::MHChain) = c.step
steps(c::MHChain) = c.steps
function increment!(c::MHChain)
    c.step += 1
    return nothing
end

function initialize_procedure(proc::MCMC,
                              query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    trace
end

function initialize_chain(proc::MCMC,
                          query::Q,
                          n::Int) where {Q<:Query}
    MHChain{Q}(query, proc, n)
end

function step!(chain::MHChain{StaticQuery})
    @unpack proc, state = chain
    chain.state = proc.update(state)
    chain.step += 1
    return nothing
end
