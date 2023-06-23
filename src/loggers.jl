using JLD2
using DataStructures

export NullLogger,
    null_logger,
    MemLogger,
    JLD2Logger

struct NullLogger <: ChainLogger
end

buffer(::NullLogger) = nothing

function report_step!(logger::NullLogger,
                      chain::InferenceChain)
    return nothing
end

mutable struct MemLogger <: ChainLogger
    buffer::CircularDeque{ChainDigest}
    MemLogger(n::Int) = new(CircularDeque{ChainDigest}(n))
end

buffer(l::MemLogger) = l.buffer

function report_step!(logger::MemLogger,
                      chain::InferenceChain)

    p = estimator(chain)
    q = estimand(chain)
    bfr = buffer(logger)
    # extract digest and push to buffer
    push!(bfr, digest(q, chain))
    return nothing
end

#TODO: handle overwritting
mutable struct JLD2Logger <: ChainLogger
    buffer::CircularDeque{ChainDigest}
    path::String
    function JLD2Logger(n::Int, p::String;
                        overwrite=false)
        new(CircularDeque{ChainDigest}(n), p)
    end
end

buffer(l::JLD2Logger) = l.buffer

function report_step!(logger::JLD2Logger,
                      chain::InferenceChain)
    p = estimator(chain)
    q = estimand(chain)
    bfr = buffer(logger)
    idx = step(chain)

    # extract digest and push to buffer
    push!(bfr, digest(q, chain))

    # determine if buffer is full
    buffer_idx = length(bfr)
    isfull = capacity(bfr) == buffer_idx

    # write if full or chain is done
    if isfull || is_finished(chain)
        @debug "writing at step $idx"
        start = idx - buffer_idx + 1
        # no path to save, exit
        jldopen(logger.path, "a+") do file
            # REVIEW: This can hang depending on chain complexity
            # save current chain
            haskey(file, "current_chain") && delete!(file, "current_chain")
            file["current_chain"] = chain
            haskey(file, "current_idx") && delete!(file, "current_idx")
            file["current_idx"] = idx
            # save digest buffer
            for j = start:idx
                file["$j"] = popfirst!(bfr)
            end
        end
    end
    return nothing
end

function resume_chain(l::ChainLogger)
    @assert isfile(l.path) "Path $path does not exist"
    chain, idx = _latest_state(path) #TODO
    run_chain!(chain, l)
    return chain
end
