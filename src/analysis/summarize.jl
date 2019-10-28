using DataFrames

export to_frame

function to_frame(results::SequentialTraceResult)

    dims = size(results.estimates)
    samples = collect(1:dims[2])
    names = [:t, :sid, results.latents..., :log_score]
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

function to_frame(results::StaticTraceResult)

    dims = size(results.estimates)
    samples = collect(1:dims[2])
    names = [:iter, :sid, results.latents..., :log_score]
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
