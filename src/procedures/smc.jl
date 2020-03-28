export sequential_monte_carlo,
    tracked_latents,
    save_state

import Base.Filesystem
using FileIO
using JLD2

abstract type SequentialChain <: InferenceChain end


function record_state!(r::SequentialChain, start::Int, stop::Int)
    if !isnothing(r.path)
        jldopen(r.path, "a+") do file
            for (i,j) = enumerate(start:stop)
                println("$i $j")
                file["state/$j"] = r.towrite_buffer[i]
            end
        end
    end
end

function resume_inference(path::String)
    error("not implemented")
end

function sequential_monte_carlo(procedure::InferenceProcedure,
                                query::SequentialQuery;
                                path::Union{String, Nothing} = nothing)
    # Initialized data structures that hold inference traces
    results = initialize_results(procedure, query, path = path)

    # Initialize inference state
    state = initialize_procedure(procedure, query)
    # Iterate across target distributions define in query
    targets = collect(query)
    for (it, target) in enumerate(targets)
        aux_state = smc_step!(state, procedure, target)
        report_step!(results, state, query, it)
        # report_aux!(results, aux_state, query, it)
    end
    return results
end
