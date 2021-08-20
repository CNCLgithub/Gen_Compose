"""
Defines the types used through the inference library
"""
module Gen_Compose

using Gen
using UnPack

# Inference chain
include("chain.jl")

# Queries
include("queries/inference_query.jl")

# Inference procedures
include("procedures/procedure.jl")

# Misc
# include("analysis/analysis.jl")


end # module
