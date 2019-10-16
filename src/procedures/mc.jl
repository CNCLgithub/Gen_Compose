
mutable struct StaticTraceResult <: InferenceResult
    latents::AbstractVector{Symbol}
    estimates::E where E<:AbstractArray{Float64}
    log_score::Vector{Float64}
    StaticTraceResult(latents, dims) = new(latents,
                                           Array{Float64}(undef,dims...),
                                           Vector{Float64}(undef, dims[1]))
end

function initialize_results(proc::InferenceProcedure,
                            query::StaticQuery,
                            iterations::Int)
    inner = initialize_results(query)
    outer = initialize_results(proc)
    dims = (iterations, outer..., inner...)
    return StaticTraceResult(query.latents, dims)
end

function static_monte_carlo(procedure::InferenceProcedure,
                            query::StaticQuery,
                            iterations::Int)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query, iterations)

    # Begin inference procedure
    let
        state = Nothing;
    end
    for it in range(1, length=iterations)
        addr = :iter => it
        if it == 1
            state = initialize_procedure(procedure, query, addr)
        else
            step_procedure!(state, procedure, query, addr)
        end

        # Report step
        report_step!(results, state, query.latents, it)
    end
    return results
end

export static_monte_carlo
