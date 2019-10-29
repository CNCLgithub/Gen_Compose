using Gadfly
using Compose

export visualize

function estimate_layer(estimate, geometry = Gadfly.Geom.hexbin)
    layer(x = :t, y = estimate, geometry)
end

function plot_map(latents, estimates, xaxis,
                  geometry = Gadfly.Geom.dot)
    plot(x=xs, y=estimates, geometry,
         Guide.xlabel(xaxis), Guide.ylabel(name))
end

"""
Returns a summary plot containing:

1. The histogram of estimates as a function of time (for each latent)
2. The histogram of log scores as a function of time
"""
function visualize(results::SequentialTraceResult)
    df = to_frame(results)
    # first the estimates
    estimates = map(estimate_layer, results.latents)
    # last log scores
    log_scores = plot_latent(:log_score)
    plot = plot(df,
                estimates...,
                log_scores)
end
