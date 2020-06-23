export AbstractParticleFitler,
    ParticleFilter

abstract type AbstractParticleFilter <: InferenceProcedure end
struct ParticleFilter <: AbstractParticleFilter
    particles::U where U<:Int
    ess::Float64
    rejuvenation::T where T<:Function
end

function rejuvenate!(proc::AbstractParticleFilter,
                     state::Gen.ParticleFilterState)
    # add rejuvenation
    for p=1:proc.particles
        state.traces[p] = proc.rejuvenation(state.traces[p])
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

function smc_step!(state::Gen.ParticleFilterState,
                   proc::AbstractParticleFilter,
                   query::StaticQuery)
    # Resample before moving on...
    resample!(proc, state)
    # update the state of the particles
    Gen.particle_filter_step!(state, query.args,
                              (UnknownChange(),),
                              query.observations)
    rejuvenate!(proc, state)
    return nothing
end


mutable struct SeqPFChain <: SequentialChain
    buffer::Vector{T} where {T}
    buffer_idx::Int
    path::Union{String, Nothing}
end

isfull(c::SeqPFChain) = c.buffer_idx == length(c.buffer)

function initialize_results(proc::AbstractParticleFilter,
                            query::SequentialQuery;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 40)

    buffer = Vector{Dict}(undef, buffer_size)
    isnothing(path) || isfile(path) && rm(path)
    return SeqPFChain(buffer, 1, path)
end
function initialize_results(proc::AbstractParticleFilter,
                            query::SequentialQuery,
                            resume::Int;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 40)

    buffer = Vector{Dict}(undef, buffer_size)
    return SeqPFChain(buffer, 1, path)
end


function report_step!(chain::SeqPFChain,
                      state::Gen.ParticleFilterState,
                      aux_state::Any,
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
        "ml_est" => log_ml_estimate(state),
        "aux_state" => aux_state
    )

    buffer = chain.buffer
    buffer[chain.buffer_idx] = step_parse

    # write buffer to disk
    isfinished = (idx == length(query))
    if isfull(chain) || isfinished
        println("writing at step $idx")
        start = idx - chain.buffer_idx + 1
        if !isnothing(chain.path)
            jldopen(chain.path, "a+") do file
                for (i,j) = enumerate(start:idx)
                    file["$j"] = chain.buffer[i]
                end
            end
        end
        buffer = isfinished ? buffer : Vector{Dict}(undef, length(chain.buffer))
        chain.buffer_idx = 1
    else
        # increment
        chain.buffer_idx += 1
    end
    chain.buffer = buffer
    return nothing
end

function resume_procedure(proc::AbstractParticleFilter,
                          query::SequentialQuery,
                          rid::Int,
                          choices::Dict)
    (rid < 1) && error("Resume index must be > 1")
    prev_target_dis = query[rid-1]
    obs = prev_target_dis.observations
    args = prev_target_dis.args
    state = initialize_procedure(proc, query)
    for i=1:proc.particles
        constraints = deepcopy(obs)
        for (k,v) in choices
            constraints[k] = v[i]
            println("$k => $(v[i])")
        end
        (state.new_traces[i], increment, _, discard) = update(
            state.traces[i], args, (UnknownChange(),), constraints)
        # if !isempty(discard)
        #     error("Choices were updated or deleted inside particle filter step: $discard")
        # end
        state.log_weights[i] += increment
    end

    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp

    return state

end
