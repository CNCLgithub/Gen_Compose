export AbstractParticleFitler,
    ParticleFilter

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

function update_chain!(c::SeqPFChain, trace)
    c.buffer[c.buffer_idx] = trace
    return nothing
end

function initialize_results(proc::AbstractParticleFilter,
                            query::SequentialQuery;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 40)

    buffer = Vector{Dict}(undef, buffer_size)
    isnothing(path) || isfile(path) && rm(path)
    return SeqPFChain(buffer, 1, path)
end

function report_step!(chain::SeqPFChain,
                      state::Gen.ParticleFilterState,
                      query::Query,
                      idx::Int)
    traces = get_traces(state)
    n = length(traces)
    w_traces = Gen.get_traces(state)
    uw_traces = Gen.sample_unweighted_traces(state, n)

    weighted = map(t -> parse_trace(query, t), w_traces)
    unweighted = map(t -> parse_trace(query, t), uw_traces)
    step_parse = Dict(
        "weighted" => merge(hcat, weighted...),
        "unweighted" => merge(hcat, unweighted...),
        "log_scores" => reshape(get_log_weights(state), (1, n)),
        "unweighted_scores" => map(get_score, uw_traces),
        "ml_est" => log_ml_estimate(state)
    )

    buffer = chain.buffer
    buffer[chain.buffer_idx] = step_parse
    # write buffer to disk
    if (chain.buffer_idx == length(chain.buffer))
        start = idx - n + 1
        if !isnothing(chain.path)
            jldopen(chain.path, "a+") do file
                for (i,j) = enumerate(start:idx)
                    file["state/$j"] = chain.buffer[i]
                end
            end
        end
        buffer = Vector{Dict}(undef, length(chain.buffer))
        chain.buffer_idx = 1
    else
        # increment
        chain.buffer_idx += 1
    end
    chain.buffer = buffer
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
