export AbstractLatentMap,
    LatentMap

"""Defines a functor from a trace -> a collection of choices"""
abstract type AbstractLatentMap end

struct LatentMap <: AbstractLatentMap
    map::Dict{Symbol, Function}
end

function Base.keys(lm::LatentMap)
    keys(lm.map)
end

function Base.length(lm::LatentMap)
    length(lm.map)
end

function (lm::LatentMap)(t::Gen.Trace)
    d = Dict{Symbol, Any}()
    for (k,f) in lm.map
        d[k] = f(t)
    end
    return d
end
