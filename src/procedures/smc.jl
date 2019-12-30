export sequential_monte_carlo

mutable struct SequentialTraceResult <: InferenceResult
    latents::T where T<:AbstractVector
    estimates::E where E<:AbstractArray{Float64}
    log_score::Array{Float64,2}
    axis::String
    SequentialTraceResult(latents, dims) = new(latents,
                                               Array{Float64}(undef, dims...),
                                               Array{Float64}(undef, dims[1:2]...),
                                               "time")
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
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query)

    # collapses the query into target distributions
    targets = collect(query)
    # Begin inference procedure
    state = initialize_procedure(procedure, targets[1])
    report_step!(results, state, 1)
    for it = 2:length(query)
        smc_step!(state, procedure, targets[it])
        report_step!(results, state, it)
    end
    return results
end

