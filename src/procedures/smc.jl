export sequential_monte_carlo,
    sequential_monte_carlo!,
    resume_chain,
    SequentialChain


abstract type SequentialChain <: InferenceChain end


function sequential_monte_carlo(procedure::InferenceProcedure,
                                query::SequentialQuery,
                                path::Union{Nothing, String},
                                buffer_size::Int64)
    # Initialized data structures that hold inference traces
    chain = initialize_chain(procedure, query)
    buffer = CircularDeque{ChainDigest}(buffer_size)
    sequential_monte_carlo!(chain, 1, buffer, path)
    return chain
end

function sequential_monte_carlo!(chain::SequentialChain,
                                 start_idx::Int64,
                                 buffer::CircularDeque{ChainDigest},
                                 path::Union{Nothing, String})
    @unpack query = chain
    # Iterate across target distributions define in query
    for it = start_idx:length(query)
        smc_step!(chain, it)
        report_step!(buffer, chain, it, path)
    end
    return nothing
end

function resume_chain(path::String, buffer_size::Int64)
    @assert isfile(path) "Path $path is not a file"
    chain, idx = load(path, "current_chain", "current_idx")
    buffer = CircularDeque{ChainDigest}(buffer_size)
    sequential_monte_carlo!(chain, idx + 1, buffer, path)
    return chain
end
