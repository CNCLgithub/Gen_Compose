export MCMC, MCMCTrace, MetropolisHastings, MHTrace, StaticMHChain

abstract type MCMC <: InferenceProcedure end

"Apply the update function to a chain state"
function update end

"""
Simple definition of MH procedure
"""
struct MetropolisHastings <: MCMC
    update::T where T<:Function
end

update(p::MetropolisHastings, tr::Gen.Trace) = p.update(tr)

"""
Static MH chain
"""
mutable struct MHChain{Q, P<:MCMC} <: InferenceChain{Q, P}
    query::Q
    proc::P
    state::Gen.Trace
    auxillary::AuxillaryState
    step::Int
    steps::Int
end

function MHChain{Q}(q::Q,
                    p::MCMC,
                    n::Int,
                    i::Int = 1) where {Q<:Query}
    state = initialize_procedure(p, q)
    aux = EmptyAuxState()
    return MHChain{Q}(q, p, state, aux, i, n)
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
