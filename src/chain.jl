export InferenceChain,
    AuxillaryState,
    EmptyAuxState,
    LatentMap,
    Digest,
    digest

"""Data defining inference chain"""
abstract type InferenceChain end

"""Auxillary state for procedure"""
abstract type AuxillaryState end

"""Dummy auxillary state"""
struct EmptyAuxState <: AuxillaryState end
