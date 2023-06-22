using Gen
using Gen_Compose

@gen function kernel(x::Float64, m::Float64, b::Float64, sigma::Float64)
    y::Float64 = @trace(normal(x*m + b, sigma), :y)
    return y
end

@gen function gm(xs::Vector{Float64})
    m = @trace(uniform(-5.0, 5.0), :m)
    b = @trace(uniform(-10., 10.), :b)
    sigma = @trace(gamma(2.0, 1.0), :sigma)
    n = length(xs)
    ms = fill(m, n)
    bs = fill(b, n)
    ss = fill(sigma, n)
    ys = @trace(Map(kernel)(xs, ms, bs, ss), :ys)
    return ys
end

@load_generated_functions

# Observations
gt_m = 2.0
gt_b = -1.0
xs = collect(LinRange(0, 5, 10))
ys = @. gt_m * xs + gt_b

obs = Gen.choicemap()
for i = 1:length(ys)
    obs[:ys => i => :y] = ys[i]
end

function lm1(c::InferenceChain)
    trace = estimate(c)
    (line = (trace[:m], trace[:b]),
     ls = get_score(trace))
end

lm = LatentMap(:line => lm1)

# Query - the estimand

query = StaticQuery(lm, gm, (xs,), obs)

# Define the inference procedure

@gen function line_proposal(current_trace)
    m ~ normal(current_trace[:m], 0.5)
    b ~ normal(current_trace[:b], 0.5)
end;

function gaussian_drift_update(tr)
    # Gaussian drift on line params
    (tr, _) = mh(tr, line_proposal, ())

    # Block resimulation: Update the outlier classifications
    (xs,) = get_args(tr)
    n = length(xs)
    for i=1:n
        (tr, _) = mh(tr, select(:ys => i => :y))
    end

    # Block resimulation: Update the prob_outlier parameter
    (tr, w) = mh(tr, select(:sigma))
    tr
end;

proc = MetropolisHastings(gaussian_drift_update)

# initialize and run the chain

nsteps = 100
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
