using Gen
using Gen_Compose

# Helper


struct RandomVec <: Gen.Distribution{Vector{Float64}} end

const random_vec = RandomVec()

function Gen.logpdf(::RandomVec, x::Vector{Float64}, mu::Vector{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end;

function Gen.random(::RandomVec, mu::Vector{U}, noise::T) where {U<:Real,T<:Real}
    vec = copy(mu)
    for i=1:length(mu)
        vec[i] = mu[i] + randn() * noise
    end
    return vec
end;
(::RandomVec)(mu, noise) = random(RandomVec(), mu, noise)

# -----------------------------------------------------------
# First define the world and the posterior
#
# The posterior to compute is P(m,b | ys)

xs = Vector{Float64}(1:10)

# The forward model
@gen function markov_model(y::Union{Nothing, Float64}, t::Int,
                           prior::DeferredPrior, addr)
    m = @trace(draw(prior, :m))
    b = @trace(draw(prior, :b))
    if typeof(y) == Nothing
        y = b
    end
    new_y = y + m
    return @trace(normal(new_y, 0.1), addr)
end


# Observations
ys = 2.5*xs + fill(10.0, length(xs))
ys = random_vec(ys, 0.1)
obs = Gen.choicemap()
Gen.set_value!(obs, :y, ys)

# define the prior over the set of latents
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
