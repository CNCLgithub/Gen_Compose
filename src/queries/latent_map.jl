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

function Base.merge(lm1::LatentMap, lm2::LatentMap)
    LatentMap(merge(lm1.map, lm2.map))
end

function (lm::LatentMap)(t::Gen.Trace)
    d = Dict{Symbol, Any}()
    for (k,f) in lm.map
        d[k] = f(t)
    end
    return d
end

function parse_trace(q::Query, trace::Gen.Trace)
    lm = latents(q)
    lm(trace)
end

function parse_trace_end(q::Query, trace::Gen.Trace)
    lm1 = latents(q)
    lm2 = latents_end(q)
    lm = merge(lm1, lm2)
    lm(trace)
end
