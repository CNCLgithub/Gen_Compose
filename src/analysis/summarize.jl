using DataFrames

export to_frame,
    extract_map

function _to_frame(results, axis)
    dims = size(results.estimates)
    samples = collect(1:dims[2])
    names = [axis, :sid, results.latents..., :log_score]
    total = DataFrame()
    for t = 1:first(dims), s = samples
        row =(t,
              s,
              results.estimates[t, s, :]...,
              results.log_score[t,s])
        column = collect(map(x -> [x],row))
        append!(total, DataFrame(column, names))
    end
    return total
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
