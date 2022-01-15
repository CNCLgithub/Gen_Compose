export static_monte_carlo,
    StaticChain

"""An inference chain for a static target distribution"""
abstract type StaticChain <: InferenceChain end


function static_monte_carlo(procedure::InferenceProcedure,
                            query::StaticQuery;
                            path::Union{String, Nothing} = nothing)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query, path = path)
    state = initialize_procedure(procedure, query)
    # report_step!(results, state, 1)

    for it in 1:procedure.samples
        mc_step!(state, procedure, query)
        report_step!(results, state, query, it)
    end
    return results
end


# TODO: generalize and properly dispath between `smc` and `mc`
# function resume_chain(path::String, buffer_size::Int64)
# end
