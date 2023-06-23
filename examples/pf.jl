using Gen
using Gen_Compose

@gen function kernel(t::Int, y_prev::Float64, xs::Vector{Float64},
                     m::Float64, b::Float64, sigma::Float64)
    xt = xs[t]
    y::Float64 = @trace(normal(xt*m + b, sigma), :y)
    return y
end

@gen function gm(n::Int, xs::Vector{Float64})
    m = @trace(uniform(-5.0, 5.0), :m)
    b = @trace(uniform(-10., 10.), :b)
    s= @trace(gamma(2.0, 1.0), :sigma)
    ys = @trace(Unfold(kernel)(n, 0.0, xs, m, b, s), :ys)
    return ys
end

@load_generated_functions

# Observations
gt_m = 2.0
gt_b = -1.0
xs = collect(LinRange(0, 5, 11))
ys = @. gt_m * xs + gt_b

obs = Vector{ChoiceMap}(undef, 10)
for i = 1:10
    cm = Gen.choicemap()
    cm[:ys => i => :y] = ys[i+1]
    obs[i] = cm
end

function best_particle(c::InferenceChain)
    pf_state = estimate(c)
    best = argmax(pf_state.log_weights)
    best_trace = pf_state.traces[best]
    (sigma = best_trace[:sigma],
     ls = get_score(best_trace))
end

lm = LatentMap(:best_particle => best_particle)

# Query - the estimand

args = [(i, xs) for i = 1:10]
argdiffs = [(UnknownChange(), NoChange()) for _ = 1:10]
query = SequentialQuery(lm, gm,
                        (0, xs),
                        choicemap(),
                        args,
                        argdiffs,
                        obs)

# Define the inference procedure

@gen function line_proposal(current_trace)
    m ~ normal(current_trace[:m], 0.5)
    b ~ normal(current_trace[:b], 0.5)
end;

function gaussian_drift_update(tr)
    # Gaussian drift on line params
    (tr, _) = mh(tr, line_proposal, ())

    # Block resimulation: Update the prob_outlier parameter
    (tr, w) = mh(tr, select(:sigma))
    tr
end;

particles = 100
ess = 0.5
proc = ParticleFilter(particles, ess, gaussian_drift_update)

# initialize and run the chain
nsteps = length(xs)
chain = run_chain(proc, query, nsteps)

# by default, `run_chain` doesn't log intermediate steps in the chain
# However, we can call `digest` on the current head of the chain.
digested = digest(query, chain)
display(digested)

# Let's use the in memory logger to get intermediate steps on a new chain.
mlog = MemLogger(nsteps)
chain = run_chain(proc, query, nsteps, mlog)
display(length(buffer(mlog)))

# We can also log to disk
dlog = JLD2Logger(nsteps, "chain_log.jld2")
chain = run_chain(proc, query, nsteps, dlog)

#TODO: resume chain
