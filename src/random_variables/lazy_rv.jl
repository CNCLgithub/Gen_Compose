
struct LazyDistribution{V, T} <: Gen.Distribution{T} \
    where {V :: Gen.Distribution{T}}
    param_func<:Function
end

function logpdf(lz::LazyDistribution{V, T}, x::T, value)
    params = lz.param_func(value)
    logpdf(::V, x, params)
end

function logpdf_grad(lz::LazyDistribution{V, T}, x::T, value)
    params = lz.param_func(value)
    logpdf_grad(::V, x, params)
end

function random(lz::LazyDistribution{V, T}, value)
    params = lz.param_func(value)
    random(::V, params...)
end

(lz::LazyDistribution{V, T}, args...) = random(lz, args...)

has_output_grad(lz::LazyDistribution{V, T}) = has_output_grad(::V)
has_argument_grads(lz::LazyDistribution{V, T}) = has_argument_grads(::V)

"""
    rv::Gen.Distribution = declare_rv()
Returns a RV that will...
"""
function defer_rv(rv<:Gen.Distribution{T}, param_func<:Function, addr::Gen.Selection) \
    => LazyDistribution{T}
    @gen function declared_rv(x)
        params = param_func(x)
        @trace(rv, params..., addr)
    end
    return declared_rv
end
