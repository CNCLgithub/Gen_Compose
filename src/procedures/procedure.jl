
"Computes a posterior query"
abstract type InferenceProcedure end
abstract type InferenceParameters{T} end

"""
    state = initialize_procedure(proc::InferenceProcedure,
                                 query::InferenceQuery{L,C,O})
"""
function initialize_procedure end

function step_procedure end


export InferenceProcedure
export initialize_procedure
export step_procedure

include("particle_filter.jl")
