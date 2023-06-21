export EmptyAuxState

"""Dummy auxillary state"""
struct EmptyAuxState <: AuxillaryState end

include("mh.jl")
include("particle_filter.jl")
