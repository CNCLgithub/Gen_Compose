export AbstractParticleFitler,
    ParticleFilter

abstract type AbstractParticleFilter <: InferenceProcedure end

struct ParticleFilter <: AbstractParticleFilter
    particles::U where U<:Int
    ess::Float64
    rejuvenation::T where T<:Function
end

#################################################################################
# Static query support
#################################################################################

function initialize_chain(proc::AbstractParticleFilter,
                          query::StaticQuery,
                          path::Union{String, Nothing},
                          buffer_size::Int64)
    state = initialize_procedure(proc, query)
    error("Not implemented (yet)")
end

function initialize_procedure(proc::AbstractParticleFilter,
                              query::StaticQuery)
    Gen.initialize_particle_filter(query.forward_function,
                                   query.args,
                                   query.observations,
                                   proc.particles)
end

function mc_step!(state::Gen.ParticleFilterState,
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
    @inbounds for p=1:particles
        state.traces[p] = rejuvenation(state.traces[p])
    end
end

function smc_step!(chain::SeqPFChain, i::Int64)
    @unpack proc, query = chain
    squery = query[i]
    smc_step!(chain, proc, squery)
    return nothing
end

function smc_step!(chain::SeqPFChain, proc::AbstractParticleFilter,
                   query::StaticQuery)
    @unpack state = chain
    @unpack args, observations = query
    # Resample before moving on...
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    # update the state of the particles
    argdiffs = (UnknownChange(), )
    println("taking step $(first(args))")
    @time Gen.particle_filter_step!(state, args, argdiffs,
                              observations)
    rejuvenate!(chain, proc)
    return nothing
end

function report_step!(buffer::CircularDeque{ChainDigest},
                      chain::SeqPFChain,
                      idx::Int,
                      path::Union{Nothing, String})

    @unpack query, state, auxillary = chain

    push!(buffer, digest(query, chain))

    buffer_idx = length(buffer)
    # write buffer to disk
    isfull = capacity(buffer) == buffer_idx
    isfinished = (idx == length(query))
    if isfull || isfinished
        @debug "writing at step $idx"
        start = idx - buffer_idx + 1
        # no path to save, exit
        isnothing(path) || jldopen(path, "a+") do file
            # save current chain
            haskey(file, "current_chain") && delete!(file, "current_chain")
            file["current_chain"] = chain
            haskey(file, "current_idx") && delete!(file, "current_idx")
            file["current_idx"] = idx
            # save digest buffer
            for j = start:idx
                file["$j"] = popfirst!(buffer)
            end
        end
    end
    return nothing
end
