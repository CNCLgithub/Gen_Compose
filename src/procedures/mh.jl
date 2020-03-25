struct MetropolisHastings <: InferenceProcedure
    update::T where T<:Function
end

mutable struct MHTrace
    current_trace::T where T<:Gen.DynamicDSLTrace
end

function initialize_procedure(proc::MetropolisHastings,
                              query::StaticQuery)
    addr = observation_address(query)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    return MHTrace(trace)
end

function mc_step!(state::MHTrace,
                  proc::MetropolisHastings,
                  query::StaticQuery)
    state.current_trace = proc.update(state.current_trace)
    return nothing
end

function report_step!(results::T where T<:InferenceChain,
                      state::MHTrace,
                      idx::Int)
    latents = results.latents
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

smc_step!() = error("unimplemented")

export MetropolisHastings
