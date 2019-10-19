
struct DynamicDistribution{T} <: Gen.Distribution{T}
    rv::Gen.Distribution{T}
    param_func::Function
end

function logpdf(lz::DynamicDistribution{T}, x::T, value) where {T}
    params = lz.param_func(value)
    logpdf(typeof(lz.rv), x, params)
end

function logpdf_grad(lz::DynamicDistribution{T}, x::T, value) where {T}
    params = lz.param_func(value)
    logpdf_grad(typeof(lz.rv), x, params)
end

function random(lz::DynamicDistribution{T}, value) where {T}
    params = lz.param_func(value)
    random(lz.rv, params...)
end


has_output_grad(lz::DynamicDistribution{T}) where {T} = has_output_grad(typeof(lz.rv))
has_argument_grads(lz::DynamicDistribution{T}) where {T} = has_argument_grads(typeof(lz.rv))

export DynamicDistribution
