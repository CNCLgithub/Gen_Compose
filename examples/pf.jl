using Gen
using Gen_Compose

# -----------------------------------------------------------
# First define the world and the posterior
#
# The posterior to compute is P(m,b | ys)

xs = range(1, stop=20)

function line_model(m,b,x)
    m*x + b
end

# The forward model + noise
@gen function noisy_line_model(c::T where T<:Gen.ChoiceMap)
    ys = Vector{Float64}(undef, length(xs))
    for x in xs
        y = line_model(c[:m], c[:b], x)
        # add noise ~ N(^y, 0.1)
        ys[x] = @trace(normal(y, 0.1), x => :y)
    end
    return ys
end


# The ground truth latents
gt = choicemap()
gt[:m] = 2.5
gt[:b] = 10

# Observations
ys = noisy_line_model(gt)

# define the prior over the set of latents
latents = map(first, collect(get_submaps_shallow(gt)))
prior = [ LazyDistribution{Float64}(uniform,  _ -> (-4, 4))
          LazyDistribution{Float64}(uniform,  _ -> (-20, 20))]


query = Gen_Compose.StaticQuery{Float64, Gen.ChoiceMap, Vector{Float64}}(
    latents,
    prior,
    gt,
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
moves = [ LazyDistribution{Float64}(uniform, x -> (x-0.05, x+0.05))
          LazyDistribution{Float64}(uniform, x -> (x-0.1, x+0.1))]
# the rejuvination will follow Gibbs sampling
rejuv = gibbs_steps(moves)

procedure = ParticleFilter(n_particles,
                           ess,
                           rejuv)


results = static_monte_carlo(procedure, query)

summarize(results)
