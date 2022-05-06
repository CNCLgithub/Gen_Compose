export static_monte_carlo,
    StaticChain

abstract type StaticChain <: InferenceChain end


function static_monte_carlo(procedure::InferenceProcedure,
                            query::StaticQuery;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 10)
    buffer = CircularDeque{ChainDigest}(buffer_size)
    chain = initialize_chain(procedure, query)
    static_monte_carlo!(chain, 1, buffer, path)
    return chain
end

function static_monte_carlo!(chain::StaticChain,
                             start_idx::Int64,
                             buffer::CircularDeque{ChainDigest},
                             path::Union{Nothing, String})
    @unpack proc, query = chain
    for it = start_idx:proc.samples
        mc_step!(chain, proc, it)
        report_step!(buffer, chain, it, path)
    end
    return nothing
end

function resume_mc_chain(path::String, buffer_size::Int64)
    @assert isfile(path) "Path $path is not a file"
    chain, idx = load(path, "current_chain", "current_idx")
    buffer = CircularDeque{ChainDigest}(buffer_size)
    static_monte_carlo!(chain, idx + 1, buffer, path)
    return chain
end
