export static_monte_carlo,
    StaticChain

abstract type StaticChain <: InferenceChain end


function static_monte_carlo(procedure::InferenceProcedure,
                            query::StaticQuery;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 10)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query, path = path,
                                 buffer_size = buffer_size)
    state = initialize_procedure(procedure, query)
    # report_step!(results, state, 1)

    for it in 1:procedure.samples
        aux_state = mc_step!(state, procedure, query)
        report_step!(results, state, aux_state, query, it)
    end
    return results
end

