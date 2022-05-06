export MCMC, MCMCTrace, MetropolisHastings, MHTrace, StaticMHChain

abstract type MCMC <: InferenceProcedure end

abstract type MCMCTrace end

"""
Simple definition of MH procedure
"""
struct MetropolisHastings <: MCMC
    samples::Int
    update::T where T<:Function
end

"""
Describes state of chain for MH
"""
mutable struct StaticMHChain <: StaticChain
    query::StaticQuery
    proc::MCMC
    state::Gen.Trace
    auxillary::AuxillaryState
end


function initialize_procedure(proc::MCMC,
                              query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    trace
end

function initialize_chain(proc::MCMC,
                          query::StaticQuery)
    trace = initialize_procedure(proc, query)
    return StaticMHChain(query, proc, trace, EmptyAuxState())
end

function mc_step!(chain::StaticMHChain,
                  proc::MetropolisHastings,
                  idx::Int)
    @unpack proc, state = chain
    chain.state = proc.update(state)
    return nothing
end


function report_step!(buffer::CircularDeque{ChainDigest},
                      chain::StaticMHChain,
                      idx::Int,
                      path::Union{Nothing, String})


    @unpack proc, query, state, auxillary = chain
    push!(buffer, digest(query, chain))

    buffer_idx = length(buffer)
    # write buffer to disk
    isfull = capacity(buffer) == buffer_idx
    isfinished = (idx == proc.samples)
    if isfull || isfinished
        @debug "writing at step $idx"
        start = idx - buffer_idx + 1
        # no path to save, exit
        isnothing(path) || jldopen(path, "a+") do file
            # save current chain
            haskey(file, "current_chain") && delete!(file, "current_chain")
            file["current_chain"] = chain
            haskey(file, "current_idx") && delete!(file, "current_idx")
            file["current_idx"] = idx
            # save digest buffer
            for j = start:idx
                file["$j"] = popfirst!(buffer)
            end
        end
    end
    return nothing
end


smc_step!() = error("unimplemented")

