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
    buffer::CircularBuffer{ChainDigest}
    MemLogger(n::Int) = new(CircularBuffer{ChainDigest}(n))
end

buffer(l::MemLogger) = l.buffer

function report_step!(logger::MemLogger,
                      chain::InferenceChain)
    q = estimand(chain)
    bfr = buffer(logger)
    # extract digest and push to buffer
    push!(bfr, digest(q, chain))
    return nothing
end

#TODO: handle overwritting
mutable struct JLD2Logger <: ChainLogger
    buffer::CircularBuffer{ChainDigest}
    path::String
    save_chain::Bool
    function JLD2Logger(n::Int, p::String;
                        overwrite=false,
                        save_chain=false)
        if isfile(p)
            if overwrite
                rm(p)
            else
                error("Chain log file exists.")
            end
        end
        new(CircularBuffer{ChainDigest}(n), p, save_chain)
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
        @show logger.path
        jldopen(logger.path, "a+") do file
            # REVIEW: This can hang depending on chain complexity
            # save current chain
            if logger.save_chain
                haskey(file, "current_chain") && delete!(file, "current_chain")
                file["current_chain"] = chain
            end

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
