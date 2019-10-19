using Gen
using Gen_Compose

# -----------------------------------------------------------
# First define the world and the posterior
#
# The posterior to compute is P(m,b | ys)

xs = Vector{Float64}(1:20)

# The forward model
@gen function noisy_line_model(prior::DeferredPrior)
    ys = Vector{Float64}(undef, length(xs))
    m = @trace(draw(prior, :m))
    b = @trace(draw(prior, :b))
    for x in xs
        y = m*x + b
        # add noise ~ N(^y, 0.1)
        ys[x] = @trace(normal(y, 0.1), x => :y)
    end
    return ys
end


# The ground truth latents
gt = choicemap()
gt[:m] = 2.5
gt[:b] = 10.0

# Observations
# ys = noisy_line_model(gt)
ys = 2.5*xs + fill(10.0, length(xs))

# define the prior over the set of latents
latents = map(first, collect(Gen.get_values_shallow(gt)))
prior = DeferredPrior(latents,
                      [StaticDistribution{Float64}(uniform, (-4, 4))
                       StaticDistribution{Float64}(uniform, (-20, 20))])
# prior = [LazyDistribution{Float64}(uniform,  _ -> (-4, 4))
#          LazyDistribution{Float64}(uniform,  _ -> (-20, 20))]


query = Gen_Compose.StaticQuery{Float64, Gen.ChoiceMap, Vector{Float64}}(
    latents,
    prior,
    noisy_line_model,
    ys)

# -----------------------------------------------------------
# Define the inference procedure
# In this case we will be using a particle filter
#
# Additionally, this will be under the Sequential Monte-Carlo
# paradigm.

n_particles = 10
ess = n_particles * 0.5
# defines the random variables used in rejuvination
moves = [DynamicDistribution{Float64}(uniform, x -> (x-0.05, x+0.05))
         DynamicDistribution{Float64}(uniform, x -> (x-0.1, x+0.1))]

# the rejuvination will follow Gibbs sampling
rejuv = gibbs_steps(moves, latents)

procedure = ParticleFilter(n_particles,
                           ess,
                           rejuv)

iterations = 100
results = static_monte_carlo(procedure, query, iterations)

# summarize(results)
