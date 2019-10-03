

# -----------------------------------------------------------
# First define the world and the posterior
#
# The posterior to compute is P(m,b | ys)

xs = range(1, 20)

function line_model(m,b,x) = m*x + b end
# The forward model + noise
@gen function noisy_line_model(c <: Gen.ChoiceMap)
    ^y = line_model(c[:m], c[:b], c[:x])
    # add noise ~ N(^y, 0.1)
    @trace(normal, (^y, 0.1), c[:x] => :y)
end


# The ground truth latents
gt = choicemap()
gt[:m] = 2.5
gt[:b] = 10

# Observations
ys = map(noisy_line_model, zip(repeat(gt),  xs))


latents = map(first, get_values_shallow(gt))
prior = [ LazyDistribution{Uniform{Float64}, Float64}( _ -> (-4, 4))
          LazyDistribution{Uniform{Float64}, Float64}( _ -> (-20, 20))]


query = SequentialQuery{Float64, Gen.ChoiceMap, Vector{Float64, 1}}(
    latents,
    prior,
    xs,
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
moves = [ LazyDistribution{Uniform, Float64}(x -> (x-0.05, x+0.05))
          LazyDistribution{Uniform, Float64}(x -> (x-0.1, x+0.1))]
# the rejuvination will follow Gibbs sampling
rejuv = gibbs_steps(moves)

procedure = ParticleFilter(n_particles,
                           ess,
                           rejuvination_move)


results = sequential_monte_carlo(procedure, query)

summarize(results)
