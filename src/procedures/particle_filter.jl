export AbstractParticleFitler,
    ParticleFilter

abstract type AbstractParticleFilter <: InferenceProcedure end

struct ParticleFilter <: AbstractParticleFilter
    particles::U where U<:Int
    ess::Float64
    rejuvenation::T where T<:Function
end

mutable struct PFChain{Q, P<:AbstractParticleFilter} <: InferenceChain{Q, P}
    query::Q
    proc::AbstractParticleFilter
    state::Gen.ParticleFilterState
    auxillary::AuxillaryState
    step::Int
    steps::Int

end


estimand(c::PFChain) = c.query
estimator(c::PFChain) = c.proc
estimate(c::PFChain) = c.state
auxillary(c::PFChain) = c.auxillary
step(c::PFChain) = c.step
steps(c::PFChain) = c.steps
function increment!(c::PFChain)
    c.step += 1
    return nothing
end

function initialize_chain(proc::P,
                          query::Q,
                          n::Int) where
    {Q<:Query, P<:AbstractParticleFilter}
    PFChain{Q, P}(query, proc, n)
end

#################################################################################
# Static query support
#################################################################################
function PFChain{Q, P}(q::Q,
                       p::P,
                       n::Int,
                       i::Int = 1) where
        {Q<:StaticQuery,  P<:AbstractParticleFilter}

    state = initialize_procedure(p, q)
    aux = EmptyAuxState()
    return PFChain{Q, P}(q, p, state, aux, i, n)
end

function initialize_procedure(proc::AbstractParticleFilter,
                              query::StaticQuery)
    Gen.initialize_particle_filter(query.forward_function,
                                   query.args,
                                   query.observations,
                                   proc.particles)
end

function step!(chain::PFChain{Q}) where {Q<:StaticQuery}
    selection = Gen.select(query.latents)
    for i=1:length(state.traces)
        (state.new_traces[i],
         state.log_weights[i]) = Gen.regenerate(state.traces[i],
                                                query.args,
                                                (Gen.NoChange(),),
                                                selection)
    end
    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return nothing
end

#################################################################################
# Sequential query support
#################################################################################

function PFChain{Q, P}(q::Q,
                       p::P,
                       n::Int,
                       i::Int = 1) where
        {Q<:SequentialQuery,  P<:AbstractParticleFilter}
    state = initialize_procedure(p, q)
    aux = EmptyAuxState()
    PFChain{Q, P}(q, p, state, aux, i, length(q))
end

function initialize_procedure(proc::AbstractParticleFilter,
                              query::SequentialQuery)
    args = initial_args(query)
    constraints = initial_constraints(query)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           args,
                                           constraints,
                                           proc.particles)
    return state
end

function rejuvenate!(chain::PFChain{Q}) where {Q<:SequentialQuery}
    @unpack particles, rejuvenation = (estimator(chain))
    state = estimate(chain)
    for p=1:particles
        state.traces[p] = rejuvenation(state.traces[p])
    end
end

function step!(chain::PFChain{Q}) where {Q<:SequentialQuery}
    @unpack query, proc, state, step = chain
    squery = query[step]
    @unpack args, observations = squery
    # Resample before moving on...
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    # update the state of the particles
    argdiffs = Tuple(fill(UnknownChange(), length(args)))
    Gen.particle_filter_step!(state, args, argdiffs,
                              observations)
    rejuvenate!(chain)
    return nothing
end
