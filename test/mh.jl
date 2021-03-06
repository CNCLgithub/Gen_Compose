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
@gen function noisy_line_model(xs::Vector{Float64})
    m = @trace(uniform(-4, 4), :m)
    b = @trace(uniform(-20, 20), :b)
    ys = m*xs + fill(b, length(xs))
    ys = @trace(random_vec(ys, 1.0), :ys)
    return ys
end


# Observations
ys = 2.5*xs + fill(10.0, length(xs))
obs = Gen.choicemap()
Gen.set_value!(obs, :ys, random_vec(ys, 0.1))

# define the prior over the set of latents
latents = [:m, :b]

query = Gen_Compose.StaticQuery(latents,
                                noisy_line_model,
                                (xs,),
                                obs)

# -----------------------------------------------------------
# Define the inference procedure
# In this case we will be using a MH

function mh_update(trace)
    (trace, _) = Gen.mh(trace, Gen.select(query.latents...))
    return trace
end
procedure = MetropolisHastings(mh_update)

iterations = 2000
static_monte_carlo(procedure, query, iterations)
@time results = static_monte_carlo(procedure, query, iterations)
println(last(sort(to_frame(results), :log_score), 10))

# summarize(results)
