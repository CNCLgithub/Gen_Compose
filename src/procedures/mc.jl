
mutable struct StaticTraceResult <: InferenceResult
    latents::T where T<:AbstractVector
    estimates::E where E<:AbstractArray{Float64}
    log_score::Array{Float64,2}
    StaticTraceResult(latents, dims) = new(latents,
                                           Array{Float64}(undef, dims...),
                                           Array{Float64}(undef, dims[1:2]...))
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
    state = initialize_procedure(procedure, query)
    report_step!(results, state, 1)

    for it in 2:iterations
        mc_step!(state, procedure, query)
        report_step!(results, state, it)
    end
    return results
end

export static_monte_carlo
