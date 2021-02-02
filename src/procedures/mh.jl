export MCMC, MCMCTrace, MetropolisHastings, MHTrace

abstract type MCMC <: InferenceProcedure end

abstract type MCMCTrace end

"""
Simple definition of MH procedure
"""
struct MetropolisHastings <: MCMC
    samples::Int
    update::T where T<:Function
end

# TODO: consider consolidating all chains to common api
mutable struct StaticMHChain <: StaticChain
    buffer::Vector{T} where {T}
    buffer_idx::Int
    path::Union{String, Nothing}
end
isfull(c::StaticMHChain) = c.buffer_idx == length(c.buffer)

"""
Describes state of trace for MH
"""
mutable struct MHTrace <: MCMCTrace
    current_trace
end



function initialize_procedure(proc::MCMC,
                              query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    return MHTrace(trace)
end

function mc_step!(state::MHTrace,
                  proc::MCMC,
                  query::StaticQuery)

    state.current_trace = proc.update(state.current_trace)
    return nothing
end


function initialize_results(proc::MCMC,
                            query::StaticQuery;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 500)

    buffer = Vector{Dict}(undef, buffer_size)
    isnothing(path) || isfile(path) && rm(path)
    return StaticMHChain(buffer, 1, path)
end

# function update_chain!(c::StaticMHChain, trace)
#     c.buffer[c.buffer_idx] = trace
#     return nothing
# end


function report_step!(chain::StaticMHChain,
                      state::MCMCTrace,
                      aux_state::Any,
                      query::StaticQuery,
                      idx::Int)
    parsed = parse_trace(query, state.current_trace)
    step_parse = Dict(
        "estimates" => parsed,
        "log_score" => get_score(state.current_trace),
        "aux_state" => aux_state
    )

    # write buffer to disk
    buffer = chain.buffer
    buffer[chain.buffer_idx] = step_parse
    # TODO allow this logic for static mh
    # isfinished = (idx == length(query))
    isfinished = true
    # if isfull(chain) || isfinished
    if isfull(chain)
        println("writing at step $idx")
        start = idx - chain.buffer_idx + 1
        if !isnothing(chain.path)
            jldopen(chain.path, "a+") do file
                for (i,j) = enumerate(start:idx)
                    file["$j"] = chain.buffer[i]
                end
            end
        end
        buffer = isfinished ? buffer : Vector{Dict}(undef, length(chain.buffer))
        chain.buffer_idx = 1
    else
        # increment
        chain.buffer_idx += 1
    end
    chain.buffer = buffer
    return nothing
end


smc_step!() = error("unimplemented")

