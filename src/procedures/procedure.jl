using FileIO
using JLD2
using Base.Filesystem
using DataStructures

""" Estimates a posterior query """
abstract type InferenceProcedure end

"""

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
