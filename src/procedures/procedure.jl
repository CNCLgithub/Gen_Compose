using FileIO
using JLD2
using Base.Filesystem
using DataStructures

"Computes a posterior query"
abstract type InferenceProcedure end

"""
    state = initialize_procedure(proc::InferenceProcedure,
                                 query::InferenceQuery{L,C,O})
"""
function initialize_chain end

function mc_step! end

function smc_step! end

function report_step! end

export InferenceProcedure,
    initialize_chain,
    mc_step!,
    smc_step!,
    report_step!

include("mc.jl")
include("smc.jl")

include("particle_filter.jl")
include("mh.jl")
