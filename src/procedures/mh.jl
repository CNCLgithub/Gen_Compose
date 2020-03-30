export MetropolisHastings

struct MetropolisHastings <: InferenceProcedure
    samples::Int
    update::T where T<:Function
end

mutable struct MHTrace
    current_trace
end

function initialize_procedure(proc::MetropolisHastings,
                              query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    return MHTrace(trace)
end

function mc_step!(state::MHTrace,
                  proc::MetropolisHastings,
                  query::StaticQuery)

    state.current_trace = proc.update(state.current_trace)
    return nothing
end

mutable struct StaticMHChain <: StaticChain
    buffer::Vector{Dict}
    buffer_idx::Int
    path::Union{String, Nothing}
end


function initialize_results(proc::MetropolisHastings,
                            query::StaticQuery;
                            path::Union{String, Nothing} = nothing,
                            buffer_size::Int = 500)

    buffer = Vector{Dict}(undef, buffer_size)
    isnothing(path) || isfile(path) && rm(path)
    return StaticMHChain(buffer, 1, path)
end

function update_chain!(c::StaticMHChain, trace)
    c.buffer[c.buffer_idx] = trace
    return nothing
end


function report_step!(chain::StaticMHChain,
                      state::MHTrace,
                      query::StaticQuery,
                      idx::Int)
    parsed = parse_trace(query, state.current_trace)
    step_parse = Dict(
        "estimates" => parsed,
        "log_score" => get_score(state.current_trace)
    )
    n = length(chain.buffer)
    buffer = chain.buffer
    buffer[chain.buffer_idx] = step_parse
    # write buffer to disk
    if (chain.buffer_idx == n)
        start = idx - n + 1
        if !isnothing(chain.path)
            jldopen(chain.path, "a+") do file
                for (i,j) = enumerate(start:idx)
                    file["state/$j"] = buffer[i]
                end
            end
        end
        buffer = Vector{Dict}(undef, n)
        chain.buffer_idx = 1
    else
        # increment
        chain.buffer_idx += 1
    end
    chain.buffer = buffer
    return nothing
end


smc_step!() = error("unimplemented")

