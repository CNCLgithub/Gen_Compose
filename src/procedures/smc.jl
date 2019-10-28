export sequential_monte_carlo

mutable struct SequentialTraceResult <: InferenceResult
    latents::T where T<:AbstractVector
    estimates::E where E<:AbstractArray{Float64}
    log_score::Array{Float64,2}
    SequentialTraceResult(latents, dims) = new(latents,
                                           Array{Float64}(undef, dims...),
                                           Array{Float64}(undef, dims[1:2]...))
end

function initialize_results(proc::InferenceProcedure,
                            query::SequentialQuery)
    inner = initialize_results(query)
    outer = initialize_results(proc)
    dims = (length(query), outer..., inner...)
    return SequentialTraceResult(query.latents, dims)
end

function sequential_monte_carlo(procedure::InferenceProcedure,
                                query::SequentialQuery)
    target_queries =  atomize(query)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query)
    root_addr = observation_address(query)

    # Begin inference procedure
    let
        state = Nothing;
        for (it, target_query) in enumerate(target_queries)
            addr = root_addr => it
            if it == 1
                state = initialize_procedure(procedure, target_query, addr)
            else
                step_procedure!(state, procedure, target_query, addr,
                                smc_step!)
            end

            # Report step
            report_step!(results, state, query.latents, it)
        end
    end
    return results
end

