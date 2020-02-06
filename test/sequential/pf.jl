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


@gen (static) function kernel(t::Int, prev_y, m, noise)
    new_y = m + prev_y
    @trace(normal(m + prev_y, noise), :y)
    return new_y
end

chain = Gen.Unfold(kernel)

@gen (static) function generative_model(T::Int)
    noise = 0.1
    m = @trace(normal(0, 4), :m)
    b = @trace(normal(0, 20), :b)
    y0 = @trace(normal(b, noise), :y0)
    states = @trace(chain(T, y0, m, noise), :chain)
    results = (y0, states)
    return results
end

Gen.load_generated_functions()

function test()
    # Observations
    xs = Vector{Float64}(1:5)
    ys = 2.5*xs + fill(10.0, length(xs))
    ys = random_vec(ys, 0.1)
    # Initial constraints for particle filter
    initial_obs = Gen.choicemap()
    initial_obs[:y0] = normal(10.0, 0.1)
    # Observations to condition sampling
    obs = Vector{Gen.ChoiceMap}(undef, length(xs))
    for i = 1:length(xs)
        cm = Gen.choicemap()
        cm[:chain => i => :y] = ys[i]
        obs[i] = cm
    end

    latents = Dict(
        :m => x -> :m,
        :b => x -> :b
    )
    args = [(Int(x),) for x in xs]

    query = Gen_Compose.SequentialQuery(latents,
                                        generative_model,
                                        (0,),
                                        initial_obs,
                                        args,
                                        obs)

    # -----------------------------------------------------------
    # Define the inference procedure
    # In this case we will be using a particle filter
    #
    # Additionally, this will be under the Sequential Monte-Carlo
    # paradigm.
    n_particles = 3
    ess = n_particles * 0.5

    # defines the random variables used in rejuvination
    function rejuv(trace)
        (trace, _) = Gen.mh(trace, Gen.select(:m, :b))
        return trace
    end

    procedure = ParticleFilter(n_particles,
                            ess,
                            rejuv)

    # first run to compile
    sequential_monte_carlo(procedure, query)
    @time results = sequential_monte_carlo(procedure, query)
    println(sort(to_frame(results), (:t, :log_score)))

end

test()
