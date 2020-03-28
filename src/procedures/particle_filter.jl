export AbstractParticleFitler,
    ParticleFilter

using Base.Filesystem
import Distributions

abstract type AbstractParticleFilter <: InferenceProcedure end
struct ParticleFilter <: AbstractParticleFilter
    particles::U where U<:Int
    ess::Float64
    rejuvination::T where T<:Function
end



function rejuvinate!(proc::AbstractParticleFilter,
                     state::Gen.ParticleFilterState)
    # add rejuvination
    for p=1:proc.particles
        state.traces[p] = proc.rejuvination(state.traces[p])
    end
end

function resample!(proc::AbstractParticleFilter,
                   state::Gen.ParticleFilterState,
                   verbose=false)
    # Resample depending on ess
    Gen.maybe_resample!(state, ess_threshold=proc.ess, verbose=verbose)
    return nothing
end

function initialize_procedure(proc::AbstractParticleFilter,
                              query::StaticQuery)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           query.args,
                                           query.observations,
                                           proc.particles)
    # rejuvinate!(proc, state)
    return state
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

function mc_step!(state::Gen.ParticleFilterState,
                  proc::AbstractParticleFilter,
                  query::StaticQuery)
    # Resample before moving on...
    resample!(proc, state)
    # update the state of the particles
    static_particle_filter_step!(state, query)
    rejuvinate!(proc, state)
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

function smc_step!(state::Gen.ParticleFilterState,
                   proc::AbstractParticleFilter,
                   query::StaticQuery)
    # Resample before moving on...
    resample!(proc, state)
    # update the state of the particles
    Gen.particle_filter_step!(state, query.args,
                              (UnknownChange(),),
                              query.observations)
    rejuvinate!(proc, state)
    return nothing
end


mutable struct SeqPFChain <: SequentialChain
    buffer::Vector{T} where {T}
    buffer_idx::Int
    path::Union{String, Nothing}
end

get_state_type(p::AbstractParticleFilter) = Gen.ParticleFilterState

function initialize_results(proc::AbstractParticleFilter,
                            query::SequentialQuery;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 40)

    buffer_type = get_state_type(proc)
    buffer = Vector{buffer_type}(undef, buffer_size)
    isnothing(path) || isfile(path) && rm(path)
    return SeqPFChain(buffer, 1, path)
end

function report_step!(results::SeqPFChain,
                      state::Gen.ParticleFilterState,
                      query::Query,
                      idx::Int)
    n = length(results.buffer)
    buffer = results.buffer
    buffer[results.buffer_idx] = (n == 1) ? state : deepcopy(state)
    # results.current_buffer[results.buffer_idx] = state

    # write buffer to disk
    if (results.buffer_idx == n)
        start = idx - n + 1
        if !isnothing(results.path)
            jldopen(results.path, "a+") do file
                for (i,j) = enumerate(start:idx)
                    # println("$i $j")
                    file["state/$j"] = buffer[i]
                end
            end
        end
        buffer_type = Gen.ParticleFilterState
        buffer = Vector{buffer_type}(undef, n)
        results.buffer_idx = 1
    else
        # increment
        results.buffer_idx += 1
    end
    results.buffer = buffer
    return nothing
end

function report_aux!(results::SeqPFChain,
                     aux_state,
                     query::Query,
                     idx::Int)
    key = "aux_state/$idx"
    record_state(results, key, aux_state)
    return nothing
end
