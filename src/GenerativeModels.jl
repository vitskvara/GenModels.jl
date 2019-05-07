module GenerativeModels

using Flux
using ValueHistories
using Adapt
using StatsBase # for samplers
using ProgressMeter
using SparseArrays
using Random
using BSON
import Base.collect

const Float = Float32

include("samplers.jl")
include("flux_utils.jl")
include("model_utils.jl")
include("ae.jl")
include("vae.jl")
include("tsvae.jl")
include("aae.jl")
include("wae.jl")

export AE, AAE, VAE, TSVAE, WAE
export ConvAE, ConvAAE, ConvVAE, ConvTSVAE, ConvWAE
export fit!

end # module
