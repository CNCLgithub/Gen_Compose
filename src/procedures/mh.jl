struct MetropolisHastings <: InferenceProcedure end

mutable struct MHTrace
    current_trace::T where T<:Gen.DynamicDSLTrace
end

function initialize_procedure(proc::MetropolisHastings,
                              query::StaticQuery,
                              addr)
    addr = observation_address(query)
    trace,_ = Gen.generate(query.forward_function,
                           (query.prior, query.args..., addr),
                           query.observations)
    return MHTrace(trace)
end

function step_procedure!(state::MHTrace,
                         proc::MetropolisHastings,
                         query::StaticQuery,
                         addr,
                         step_func)
    selection = Gen.select(query.latents...)
    state.current_trace, accepted = mc_step!(state, selection)
    return nothing
end

function report_step!(results::T where T<:InferenceResult,
                      state::MHTrace,
                      latents::Vector,
                      idx::Int)
    # copy log scores
    trace = state.current_trace
    results.log_score[idx] = Gen.get_score(trace)
    choices = Gen.get_choices(trace)
    for l = 1:length(latents)
        results.estimates[idx,1, l] = choices[latents[l]]
    end
    return nothing
end

initialize_results(::MetropolisHastings) = (1,)

mc_step!(state, selection) = Gen.mh(state.current_trace, selection)

export MetropolisHastings
