import Distributions

struct ParticleFilter <: InferenceProcedure
    particles::U where U<:Int
    ess::Float64
    rejuvination::T where T<:Function
end

"""

Helper that
"""
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
    # num_particles = length(state.traces)
    # (log_total_weight, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
    # ess = Gen.effective_sample_size(log_normalized_weights)
    # do_resample = ess < proc.ess
    # if do_resample
    #     weights = exp.(log_normalized_weights)
    #     Distributions.rand!(Distributions.Categorical(weights / sum(weights)), state.parents)
    #     state.log_ml_est += log_total_weight - log(num_particles)
    #     println(state.parents)
    #     for i=1:num_particles
    #         state.new_traces[i] = state.traces[state.parents[i]]
    #         state.log_weights[i] = state.log_weights[state.parents[i]]
    #     end
    #     # swap references
    #     tmp = state.traces
    #     state.traces = state.new_traces
    #     state.new_traces = tmp
    # end
    # println(do_resample)

    return nothing
end

function initialize_procedure(proc::ParticleFilter,
                              query::StaticQuery,
                              addr)
    sub_addr = observation_address(query)
    obs = choicemap()
    set_value!(obs, addr, get_value(query.observations, sub_addr))
    state = Gen.initialize_particle_filter(query.forward_function,
                                           (query.args..., query.prior, addr),
                                           obs,
                                           proc.particles)
    rejuvinate!(proc, state)
    return state
end


function step_procedure!(state,
                         proc::ParticleFilter,
                         query::StaticQuery,
                         addr,
                         step_fun)
    # Resample before moving on...
    resample!(proc, state)
    sub_addr = observation_address(query)
    obs = choicemap()
    set_value!(obs, addr, get_value(query.observations, sub_addr))
    # update the state of the particles with the new observation
    step_fun(state,
             (query.args..., query.prior, addr),
             (UnknownChange(),),
             obs)
    rejuvinate!(proc, state)
    return nothing
end


initialize_results(proc::ParticleFilter) = (proc.particles,)

function report_step!(results::T where T<:InferenceResult,
                      state::Gen.ParticleFilterState,
                      latents::Vector,
                      idx::Int)
    # copy log scores
    # retrieve estimates
    traces = Gen.get_traces(state)
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

mc_step!(state, args, args_change, obs) = Gen.particle_filter_step!(state, args,
                                                                    args_change, obs)

function smc_step!(state, args, args_change, obs)
    for i=1:length(state.traces)
        # include the current state
        new_args = (Gen.get_retval(state.traces[i]), Base.tail(args)...)
        (state.new_traces[i], weight) = Gen.update(state.traces[i],
                                                   new_args, args_change, obs)
        state.log_weights[i] += weight
    end
    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return nothing
end

export ParticleFilter


