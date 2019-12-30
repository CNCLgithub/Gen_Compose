import Distributions

struct ParticleFilter <: InferenceProcedure
    particles::U where U<:Int
    ess::Float64
    rejuvination::T where T<:Function
end

function rejuvinate!(proc::ParticleFilter,
                     state::Gen.ParticleFilterState)
    # add rejuvination
    for p=1:proc.particles
        state.traces[p] = proc.rejuvination(state.traces[p])
    end
end
function resample!(proc::ParticleFilter,
                   state::Gen.ParticleFilterState)
    # Resample depending on ess
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    return nothing
end

function initialize_procedure(proc::ParticleFilter,
                              query::StaticQuery)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           query.args,
                                           query.observations,
                                           proc.particles)
    rejuvinate!(proc, state)
    return state
end


function mc_step!(state::Gen.ParticleFilterState,
                  proc::ParticleFilter,
                  query::StaticQuery)
    # Resample before moving on...
    resample!(proc, state)
    # update the state of the particles
    static_step!(state, query)
    rejuvinate!(proc, state)
    return nothing
end


initialize_results(proc::ParticleFilter) = (proc.particles,)

function report_step!(results::T where T<:InferenceResult,
                      state::Gen.ParticleFilterState,
                      idx::Int)
    # copy log scores
    # retrieve estimates
    latents = results.latents
    # traces = Gen.get_traces(state)
    traces = Gen.sample_unweighted_traces(state, length(state.traces))
    for (t_i,trace) in enumerate(traces)
        results.log_score[idx, t_i] = Gen.get_score(trace)
        choices = Gen.get_choices(trace)
        for l = 1:length(latents)
            results.estimates[idx, t_i, l] = choices[latents[l]]
        end
    end
    return nothing
end

function extract_state(state::Gen.ParticleFilterState)
    map(Gen.get_retval, state.traces)
end

function static_step!(state::Gen.ParticleFilterState,
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
                   proc::ParticleFilter,
                   query::StaticQuery)
    # Resample before moving on...
    resample!(proc, state)
    # update the state of the particles
    sequential_step!(state, query)
    rejuvinate!(proc, state)
    return nothing
end

function sequential_step!(state::Gen.ParticleFilterState,
                          query::StaticQuery)
    for i=1:length(state.traces)
        # include the current state
        prev_state = Gen.get_retval(state.traces[i])
        new_args = (prev_state, Base.tail(query.args)...)
        (state.new_traces[i], weight) = Gen.update(state.traces[i],
                                                   new_args,
                                                   (UnknownChange(),),
                                                   query.observations)
        state.log_weights[i] += weight
    end
    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return nothing
end

export ParticleFilter
