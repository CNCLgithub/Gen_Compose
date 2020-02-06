export sequential_monte_carlo,
    tracked_latents

mutable struct SequentialTraceResult <: InferenceResult
    estimates::Dict{Symbol, Array{Float64}}
    log_score::Array{Float64,2}
    axis::String
end

function initialize_results(proc::InferenceProcedure,
                            query::SequentialQuery)
    # inner = initialize_results(query)
    t = length(query)
    dims = (t, initialize_results(proc)...)
    ests = Dict([(l, Array{Float64}(undef, dims...))
                 for l in keys(query.latents)])
    lgs = Matrix{Float64}(undef, dims...)
    return SequentialTraceResult(ests, lgs, "time")
end

tracked_latents(r::SequentialTraceResult) = keys(r.estimates)

function sequential_monte_carlo(procedure::InferenceProcedure,
                                query::SequentialQuery)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query)

    # Initialize inference state
    state = initialize_procedure(procedure, query)
    # Iterate across target distributions define in query
    targets = collect(query)
    for (it, target) in enumerate(targets)
        smc_step!(state, procedure, target)
        report_step!(results, state, query, it)
    end
    return results
end

