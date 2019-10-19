import Gen: random, logpdf

export StaticDistribution,
    random,
    logpdf

struct StaticDistribution{T} <: Gen.Distribution{T}
    rv::Gen.Distribution{T}
    params::Tuple
end

function logpdf(lz::StaticDistribution{T}, x::T) where {T}
    logpdf(lz.rv, x, lz.params...)
end

function logpdf_grad(lz::StaticDistribution{T}, x::T) where {T}
    logpdf_grad(lz.rv, x, lz.params...)
end

function random(lz::StaticDistribution{T}) where {T}
    random(lz.rv, lz.params...)
end


has_output_grad(lz::StaticDistribution{T}) where {T} = has_output_grad(typeof(lz.rv))
has_argument_grads(lz::StaticDistribution{T}) where {T} = has_argument_grads(typeof(lz.rv))
