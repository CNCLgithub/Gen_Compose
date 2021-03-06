using FileIO
using JLD2
using Base.Filesystem

"Computes a posterior query"
abstract type InferenceProcedure end
abstract type InferenceChain end

"""
    state = initialize_procedure(proc::InferenceProcedure,
                                 query::InferenceQuery{L,C,O})
"""
function initialize_procedure end

function mc_step! end

function smc_step! end

"""
The result frame is defined both by the outer loop (the inference procedure)
and the inner loop (the query)
"""
function initialize_results end

function report_step! end

export InferenceProcedure
export initialize_procedure
export step_procedure
export initialize_results
export report_step!

include("mc.jl")
include("smc.jl")

include("particle_filter.jl")
include("mh.jl")
