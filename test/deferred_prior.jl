using Gen
using Gen_Compose

# The ground truth latents
gt = choicemap()
gt[:m] = 2.5
gt[:b] = 10.0

# define the prior over the set of latents
latents = map(first, collect(Gen.get_values_shallow(gt)))
prior = DeferredPrior(latents,
                      [StaticDistribution{Float64}(uniform, (-4., 4,))
                       StaticDistribution{Float64}(uniform, (-20., 20.))])


trace,_ = generate(draw, (prior, :m))

@gen function f()
    m = @trace(draw(prior, :b))
end
trace,_ = generate(f, tuple())
