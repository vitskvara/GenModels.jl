using GenerativeModels
using Test
using Random
using Pkg

@testset "GenerativeModels" begin

@testset "utilities" begin
	@info "Testing utilities"
	include("samplers.jl")
	include("flux_utils.jl")
	include("model_utils.jl")
end

@testset "Models" begin
	@info "Testing models"
	include("ae.jl")
	include("vae.jl")
	include("tsvae.jl")
	include("aae.jl")
	include("wae.jl")
end

if "CuArrays" in keys(Pkg.installed())
	@testset "GPU support" begin
		@info "Testing GPU support"
		include("gpu.jl")
	end
end

end