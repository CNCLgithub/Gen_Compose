struct MetropolisHastings <: InferenceProcedure end

function initialize_procedure(proc::MetropolisHastings,
                              query::StaticQuery,
                              addr)
    addr = observation_address(query)
    trace,_ = Gen.generate(query.forward_function,
                           (query.prior, query.args..., addr),
                           query.observations)
    return trace
end

function step_procedure!(state,
                         proc::MetropolisHastings,
                         query::StaticQuery,
                         addr,
                         step_func)
    selection = Gen.select(query.latents)
    trace, _ = mc_step!(state, selection)
    return trace
end

function report_step!(results::T where T<:InferenceResult,
                      trace::T where T<:Gen.DynamicDSLTrace,
                      latents::Vector,
                      idx::Int)
    # copy log scores
    results.log_score[idx] = Gen.get_score(trace)
    choices = Gen.get_choices(trace)
    for l = 1:length(latents)
        results.estimates[idx,1, l] = choices[latents[l]]
    end
    return nothing
end

initialize_results(::MetropolisHastings) = (1,)

mc_step!(state, selection) = Gen.mh(state, selection)

export MetropolisHastings
