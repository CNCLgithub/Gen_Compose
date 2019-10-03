
function sequential_monte_carlo(procedure::InferenceProcedure,
                                params::InferenceParameters,
                                query::SequentialQuery)
    # Initialized data structures that hold inference traces
    results = initialize_results(params)

    # Begin inference procedure
    let
        state = Nothing;
    end
    for (it, obs) in enumerate(query.observations)
        cur_obs = choicemap()
        set_submap!(cur_obs, :obs => it, obs)
        query_slice = Query(query.latents,
                            query.distributions,
                            query.context,
                            query.forward_function,
                            cur_obs)
        if it == 1
            state = intialize_procedure(procedure, query_slice)
        else
            step_procedure!(state, procedure, query_slice)
        end

        # Report step
        @printf "Iteration %d / %d\n" it params.steps
        report_step!(results, state, it)
    end
    return results
end

# TODO: re-implement later

# function sequential_monte_carlo(procedure::InferenceProcedure,
#                                 params::InferenceParameters,
#                                 query::Query,
#                                 state = Nothing)
#     # Initialized data structures that hold inference traces
#     results = initialize_results(params)

#     # Begin inference procedure
#     for (it, obs) in enumerate(query.observations)

#         step_procedure!(state, procedure, query)

#         # Report step
#         @printf "Iteration %d / %d\n" it params.steps
#         report_step!(results, state, addrs, it)
#     end
#     return results

# end
