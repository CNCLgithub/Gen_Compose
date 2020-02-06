using Base.Iterators
using DataFrames

export to_frame,
    extract_map

function _to_frame(results, axis)

    latents = tracked_latents(results)
    dims = size(results.log_score)
    samples = collect(1:dims[1])
    df = DataFrame()
    df[axis] = repeat(samples, inner = dims[2])
    df[:sid] = repeat(collect(1:dims[2]), dims[1])

    for l in latents
        df[l] = collect(Base.Iterators.flatten(results.estimates[l]'))
    end
    df[:log_score] = collect(Base.Iterators.flatten(results.log_score'))
    return df
end

function to_frame(results::SequentialTraceResult)
    _to_frame(results, :t)
end

function to_frame(results::StaticTraceResult)
    _to_frame(results, :iter)
end

function extract_map(df::DataFrame, latents)
    by(df, :t, df ->  DataFrame(df[argmax(df[:, :log_score]),
                                   [:log_score, latents...]]))
end
