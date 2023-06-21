using Gen
using Gen_Compose

@gen function kernel(t::Int, x::Float64, sigma::Float64)
    y::Float64 = @trace(normal(x, sigma), :y)
    return y
end

@gen function seq_gm(n::Int)
    m = @trace(uniform(-5.0, 5.0), :m)
    b = @trace(uniform(-10., 10.), :b)
    sigma = @trace(gamma(2.0, 1.0), :sigma)
    x0 = @trace(normal(0., sigma), :x0)
    ys = @trace(Unfold(kernel)(n, x0, sigma), :ys)
    return ys
end

@load_generated_functions

# Observations
gt_m = 2.0
gt_b = -1.0
xs = LinRange(0, 5, 10)
ys = @. gt_m * xs + gt_b

obs = Gen.choicemap()
for i = 1:length(ys)
    obs[:ys => i => :y] = ys[i]
end

# Query - the estimand

query = SequentialQuery(lm, gm, (n,), )

latents = [:m, :b]
prior = DeferredPrior(latents,
                      [StaticDistribution{Float64}(uniform, (-4, 4))
                       StaticDistribution{Float64}(uniform, (-20, 20))])


query = Gen_Compose.SequentialQuery(latents,
                                    prior,
                                    markov_model,
                                    tuple(),
                                    obs)

# -----------------------------------------------------------
# Define the inference procedure
# Define the inference procedure
# In this case we will be using a particle filter
#
# Additionally, this will be under the Sequential Monte-Carlo
# paradigm.
n_particles = 100
ess = n_particles * 0.5
# defines the random variables used in rejuvination
moves = [DynamicDistribution{Float64}(uniform, x -> (x-0.05, x+0.05))
         DynamicDistribution{Float64}(uniform, x -> (x-0.1, x+0.1))]

# the rejuvination will follow Gibbs sampling
rejuv = gibbs_steps(moves, latents)

procedure = ParticleFilter(n_particles,
                           ess,
                           rejuv)

results = sequential_monte_carlo(procedure, query)
println(to_frame(results))

using Gadfly
function estimate_layer(estimate, geometry = Gadfly.Geom.histogram2d)
    layer(x = :t, y = estimate, geometry)
end


"""
Returns a summary plot containing:

1. The histogram of estimates as a function of time (for each latent)
2. The histogram of log scores as a function of time
"""
function Gen_Compose.visualize(results::Gen_Compose.SequentialTraceResult)
    df = to_frame(results)
    # first the estimates
    estimates = map(x -> Gadfly.plot(df, estimate_layer(x), Gadfly.Coord.cartesian(xmin = 1,
                                                                                   xmax = 10)),
                    results.latents)
    # last log scores
    log_scores = Gadfly.plot(df, estimate_layer(:log_score))
    plot = vstack(estimates..., log_scores)
end
plot = visualize(results)
plot |> SVG("smc-pf.svg",30cm, 30cm)
