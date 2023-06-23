# Gen Compose

[Gen.jl](gen.dev) provides an excelent ecosystem for probablistic programming, enabling a unified language for generative models and infernce procedures. However, using Gen to develop and analyze combinations of posteriors and inference procedures requires a considerable amount of boilerplait. `Gen_Compose` aims to standardize inference approximation without limiting the expressivity of Gen.  

## Design Philosophy

`Gen_Compose` revolves around `InferenceChain{Q,P}` which is parameterized by a combination of `Q<:Query` and `P<:InferenceProcedure`.  A query denotes an approximation target (e.g., a conditional distribution / posterior over a generative model) or more formally as an estimand. A inference procedure serves as an estimator over the estimand. The result of this combination is the `InferenceChain` (the estimate). 

``` julia-repl
julia> run_chain(pf, seq_query, nsteps)
PFChain{SequentialQuery, ParticleFilter}
```

The queries and procedures provided by `Gen_Compose` are not intended to be exhaustive, but instead provide a template for trivial extensability. For example, a more complex instance of `AbstractParticleFilter` or `MCMC` could be implemented by adding a new `step!` or `initialize_procedure` method. 

`Gen_Compose` provides two basic forms of queries: 

1. `StaticQuery` : an unfactorized conditional of the form $Pr(H \mid O)$
2. `SequentialQuery` : an time-factorized conditional of the form $Pr(H \mid O) \sym Pr(H_0) * \prod_t Pr(O_t \mid H_t) Pr(H_t \mid H_{t-1})$

Like procedures, new queries could be defined that explore different forms of factorization over $Pr(H \mid O)$. 
It's straightforward to extend a procedure to take advantage of such factorization by simply defining a method such as 

``` julia
step!(chain::PFChain{Q}) where {Q<:MyQuery}
```

If one requires a more extensive overhaul, then defining new subtypes of `InferenceChain{Q, P}` could look like 

``` julia
abstract type MyParticleFilter <: AbstractParticleFilter end
mutable struct MyPFChain{Q<:MyQuery, P<:MyParticleFilter} <: InferenceChain{Q, P}
    ...
end
```

## Chain Logging

``` julia
dlog = JLD2Logger(nsteps, "chain_log.jld2";overwrite = true)
chain = run_chain(proc, query, nsteps, dlog)
```

