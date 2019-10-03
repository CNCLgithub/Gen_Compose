
struct ParticleFilter <: InferenceProcedure
    particles:<Int
    ess:<Float
    rejuvination::RejuvinationMove
end



"""
Perturb each latent sequentially
using `trunc_norm_perturb`
"""
function gen_gibbs_trunc_norm(latents<:AbstractVector{Gen.Selection},
                              rv_params<:AbstractVector{Gen.Distribution{Any}})
    n_latents = length(latents)
    blocks = mh_rejuvenate(repeat([trunc_norm_perturb], n_latents))
    return trace -> blocks(trace, zip(latents, rv_params))
end;

"""

Helper that
"""
function refine_and_resample!(proc::PartcileFilter,
                             state)
    # add rejuvination
    for p=1:params.n_particles
        state.traces[p] = proc.rejuvination(state.traces[p])
    end
    # Resample depending on ess
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    return state
end

function initialize_procedure(proc::ParticleFilter,
                              query::Query{L,C,O})
    state = Gen.initialize_particle_filter(sample,
                                           (query,),
                                           proc.particles)
    state = refine_and_resample(proc, state)
end

function step_procedure!(proc::ParticleFilter,
                        query::Query{L,C,O},
                        state)
    # update the state of the particles with the new observation
    Gen.particle_filter_step!(state,
                              (query,),
                              (UnknownChange(),),
                              cur_obs)
    state = refine_and_resample!(proc, state)
end
