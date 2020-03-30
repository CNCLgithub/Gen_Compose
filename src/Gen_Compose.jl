"""
Defines the types used through the inference library
"""
module Gen_Compose
using Gen

# Random Variables
include("random_variables/random_variables.jl")

# Queries
include("queries/inference_query.jl")

# Inference procedures
include("procedures/procedure.jl")

# Misc
# include("perturbations/perturb.jl")
# include("analysis/analysis.jl")


end # module
