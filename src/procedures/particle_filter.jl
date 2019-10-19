struct ParticleFilter <: InferenceProcedure
    particles::U where U<:Int
    ess::Float64
    rejuvination::T where T<:Function
end


# function _create_result()

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
                              query::Query{L,C,O} where {L,C,O},
                              addr)
    obs = choicemap()
    set_submap!(obs, addr, query.observations)
    # state = Gen.initialize_particle_filter(sample,
    #                                        (query, addr),
    #                                        obs,
    #                                        proc.particles)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           (addr,),
                                           obs,
                                           proc.particles)
    refine_and_resample!(proc, state)
    return state
end

function step_procedure!(state,
                         proc::ParticleFilter,
                         query::Query{L,C,O} where {L,C,O},
                         addr)
    obs = choicemap()
    set_submap!(obs, addr, query.observations)
    # update the state of the particles with the new observation
    Gen.particle_filter_step!(state,
                              (query, addr),
                              (UnknownChange(),),
                              obs)
    refine_and_resample!(proc, state)
    return nothing
end

initialize_results(proc::ParticleFilter) = (proc.particles,)

function report_step!(results::T where T<:InferenceResult,
                      state::Gen.ParticleFilterState,
                      latents::Vector{Gen.Selection},
                      idx::Int)
    # copy log scores
    println(size(Gen.get_log_weights(state)))
    println(size(results.log_score))
    results.log_score[idx, :] = Gen.get_log_weights(state)
    # retrieve estimates
    traces = Gen.get_traces(state)
    for t in traces
        choices = Gen.get_choices(state.traces[t])
        for l = 1:length(latents)
            results.estimates[idx, t, l] = choices[latents[l]]
        end
    end
end


export ParticleFilter


