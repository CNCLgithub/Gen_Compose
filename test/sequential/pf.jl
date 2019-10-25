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
@gen function markov_model(prior::DeferredPrior, t::Int, addr)
    m = @trace(draw(prior, :m))
    b = @trace(draw(prior, :b))
    y = m*t + b
    y = @trace(normal(y, 0.1), addr)
    return y
end


# Observations
ys = 2.5*xs + fill(10.0, length(xs))
obs = Gen.choicemap()
Gen.set_value!(obs, :obs, random_vec(ys, 0.1))

# define the prior over the set of latents
latents = [:m, :b]
prior = DeferredPrior(latents,
                      [StaticDistribution{Float64}(uniform, (-4, 4))
                       StaticDistribution{Float64}(uniform, (-20, 20))])


query = Gen_Compose.SequentialQuery(latents,
                                    prior,
                                    noisy_line_model,
                                    tuple(),
                                    obs)

# -----------------------------------------------------------
# Define the inference procedure
# In this case we will be using a MH

procedure = MetropolisHastings()

iterations = 100
results = static_monte_carlo(procedure, query, iterations)

# summarize(results)
