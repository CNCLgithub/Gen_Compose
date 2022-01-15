export InferenceChain,
    AuxillaryState,
    EmptyAuxState

"""Data defining an inference chain"""
abstract type InferenceChain end

"""Auxillary state for procedure"""
abstract type AuxillaryState end

"""Dummy auxillary state"""
struct EmptyAuxState <: AuxillaryState end
