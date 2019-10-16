
mutable struct StaticTraceResult{L} <: InferenceResult
    latents::AbstractVector{Symbol}
    estimates::E where E<:AbstractArray{L}
    log_score::Vector{Float64}
    StaticTraceResult{L}(latents, dims) = new(latents,
                                              new Array{L}(undef,dims...),
                                              new Vector{Float64}(undef, dims[1]))
end

function initialize_results(proc::InferenceProcedure,
                            query::StaticQuery,
                            iterations::Int)
    inner = initialize_results(query)
    outer = initialize_results(proc, inner)
    dims = (iterations, outer..., inner...)
    return new StaticTraceResult{Float64}(query.latents, dims)

function static_monte_carlo(procedure::InferenceProcedure,
                            query::StaticQuery,
                            iterations::Int)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query, iterations)

    # Begin inference procedure
    let
        state = Nothing;
    end
    for it in range(iterations)
        addr = :iter => it
        cur_obs = choicemap()
        set_submap!(cur_obs, addr, query.observations)
        if it == 1
            state = intialize_procedure(procedure, query, addr)
        else
            step_procedure!(state, procedure, query, addr)
        end

        # Report step
        @printf "Iteration %d" it
        report_step!(results, state, query.latents, it)
    end
    return results
end

export static_monte_carlo
