export sequential_monte_carlo,
    SequentialChain


abstract type SequentialChain <: InferenceChain end



function resume_inference(path::String)
    error("not implemented")
end

function sequential_monte_carlo(procedure::InferenceProcedure,
                                query::SequentialQuery;
                                path::Union{String, Nothing} = nothing)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query, path = path)

    # Initialize inference state
    state = initialize_procedure(procedure, query)
    # Iterate across target distributions define in query
    targets = collect(query)
    for (it, target) in enumerate(targets)
        aux_state = smc_step!(state, procedure, target)
        report_step!(results, state, target, it)
        # report_aux!(results, aux_state, query, it)
    end
    return results
end
