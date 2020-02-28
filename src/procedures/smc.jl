export sequential_monte_carlo,
    tracked_latents,
    save_state

import Base.Filesystem
using FileIO
using JLD2

mutable struct SequentialTraceResult <: InferenceResult
    path::String
    io::JLD2.JLDFile
end

function initialize_results(proc::InferenceProcedure,
                            query::SequentialQuery)
    # inner = initialize_results(query)
    (path, _) = Base.Filesystem.mktemp("/dev/shm", cleanup = true)

    io = jldopen(path, "w")
    io["query"] = query
    io["procedure"] = proc
    return SequentialTraceResult(path, io)
end

function record_state(r::SequentialTraceResult, key, state)
    # io = jldopen(r.path, "a+")
    r.io[key] = state
    return nothing
end

function save_state(r::SequentialTraceResult, path::String)
    close(r.io)
    Base.Filesystem.cp(r.path, path, force = true)
end

function resume_inference(path::String)
end

function sequential_monte_carlo(procedure::InferenceProcedure,
                                query::SequentialQuery)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query)

    # Initialize inference state
    state = initialize_procedure(procedure, query)
    # Iterate across target distributions define in query
    targets = collect(query)
    for (it, target) in enumerate(targets)
        aux_state = smc_step!(state, procedure, target)
        report_step!(results, state, query, it)
        report_aux!(results, aux_state, query, it)
    end
    return results
end
