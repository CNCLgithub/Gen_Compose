struct ParticleFilter <: InferenceProcedure
    particles::U where U<:Int
    ess::Float64
    rejuvination::T where T<:Function
end

"""

Helper that
"""
function refine_and_resample!(proc::ParticleFilter,
                              state::Gen.ParticleFilterState)
    # add rejuvination
    for p=1:proc.particles
        state.traces[p] = proc.rejuvination(state.traces[p])
    end
    # Resample depending on ess
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    return nothing
end

function initialize_procedure(proc::ParticleFilter,
                              query::StaticQuery,
                              addr)
    sub_addr = observation_address(query)
    obs = choicemap()
    set_value!(obs, addr, get_value(query.observations, sub_addr))
    state = Gen.initialize_particle_filter(query.forward_function,
                                           (query.prior, query.args..., addr),
                                           obs,
                                           proc.particles)
    refine_and_resample!(proc, state)
    return state
end


function step_procedure!(state,
                         proc::ParticleFilter,
                         query::StaticQuery,
                         addr)
    sub_addr = observation_address(query)
    obs = choicemap()
    set_value!(obs, addr, get_value(query.observations, sub_addr))
    # update the state of the particles with the new observation
    Gen.particle_filter_step!(state,
                              (query.prior, query.args..., addr),
                              (UnknownChange(),),
                              obs)
    refine_and_resample!(proc, state)
    return nothing
end

initialize_results(proc::ParticleFilter) = (proc.particles,)

function report_step!(results::T where T<:InferenceResult,
                      state::Gen.ParticleFilterState,
                      latents::Vector,
                      idx::Int)
    # copy log scores
    results.log_score[idx, :] = Gen.get_log_weights(state)
    # retrieve estimates
    traces = Gen.get_traces(state)
    for (t_i,trace) in enumerate(traces)
        choices = Gen.get_choices(trace)
        for l = 1:length(latents)
            results.estimates[idx, t_i, l] = choices[latents[l]]
        end
    end
    return nothing
end


export ParticleFilter


