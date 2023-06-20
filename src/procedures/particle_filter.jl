export AbstractParticleFitler,
    ParticleFilter

abstract type AbstractParticleFilter <: InferenceProcedure end

struct ParticleFilter <: AbstractParticleFilter
    particles::U where U<:Int
    ess::Float64
    rejuvenation::T where T<:Function
end

# REVIEW: has different meaning than MH.steps
steps(p::ParticleFilter) = p.particles

#################################################################################
# Static query support
#################################################################################

mutable struct StaticPFChain <: StaticChain
    query::StaticQuery
    proc::AbstractParticleFilter
    state::Gen.ParticleFilterState
    auxillary::AuxillaryState
end

estimand(c::StaticPFChain) = c.query
estimator(c::StaticPFChain) = c.proc
estiamte(c::StaticPFChain) = c.state


function initialize_chain(proc::AbstractParticleFilter,
                          query::StaticQuery)
    state = initialize_procedure(proc, query)
    aux = EmptyAuxState()
    return StatPFChain(query, proc, state, aux)
end

function initialize_procedure(proc::AbstractParticleFilter,
                              query::StaticQuery)
    Gen.initialize_particle_filter(query.forward_function,
                                   query.args,
                                   query.observations,
                                   proc.particles)
end

function pfstep!(state::Gen.ParticleFilterState,
               proc::AbstractParticleFilter,
               query::StaticQuery)
    # Resample before moving on...
    resample!(proc, state)
    # update the state of the particles
    static_particle_filter_step!(state, query)
    rejuvenate!(proc, state)
    return nothing
end

function static_particle_filter_step!(state::Gen.ParticleFilterState,
                                      query::StaticQuery)
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

mutable struct SeqPFChain <: SequentialChain
    query::SequentialQuery
    proc::AbstractParticleFilter
    state::Gen.ParticleFilterState
    auxillary::AuxillaryState
end


estimand(c::SeqPFChain) = c.query
estimator(c::SeqPFChain) = c.proc
estiamte(c::SeqPFChain) = c.state

function initialize_chain(proc::AbstractParticleFilter,
                          query::SequentialQuery)
    state = initialize_procedure(proc, query)
    aux = EmptyAuxState()
    return SeqPFChain(query, proc, state, aux)
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

function rejuvenate!(chain::SeqPFChain,
                     proc::ParticleFilter)
    @unpack particles, rejuvination = proc
    for p=1:particles
        state.traces[p] = rejuvenation(state.traces[p])
    end
end

function step!(c::SeqPFChain, i::Int)
    step!(c, estimator(c), i)
end

function step!(chain::SeqPFChain,
               proc::AbstractParticleFilter,
               i::Int64)
    @unpack query, state = chain
    squery = query[i]
    @unpack args, observations = squery
    # Resample before moving on...
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    # update the state of the particles
    argdiffs = Tuple([UnknownChange() for _ in args])
    Gen.particle_filter_step!(state, args, argdiffs,
                              observations)
    rejuvenate!(chain, proc)
    return nothing
end
