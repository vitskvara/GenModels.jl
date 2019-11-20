"""
	VAMP

Vamp prior with K components.
"""
struct VAMP
	K::Int
	pseudoinputs::AbstractArray

	VAMP(K::Int, xdim::Union{Tuple, Int}) = new(K, Flux.param(randn(Float32, xdim..., K)))
	VAMP(K::Int, X::AbstractArray) = new(K, Flux.Tracker.istracked(X) ? X : Flux.param(X))
end

Flux.@treelike VAMP

function sampleVamp(v::VAMP, n::Int)
	ids = rand(1:v.K, n)
	return _sampleVamp(v, ids)
end

function sampleVamp(v::VAMP, n::Int, k::Int)
	(k>v.K) ? error("Requested component id $k is larger than number of available components $(v.K)") : nothing
	ids = repeat([k], n)
	return _sampleVamp(v, ids)
end

function _sampleVamp(v::VAMP, ids::Vector)
	v.pseudoinputs[repeat([:], ndims(v.pseudoinputs)-1)..., ids]
end

# no sampling here, it should be done by mapping if necessary
function encodeSampleVamp(v::VAMP, mapping, args...)
	x = sampleVamp(v, args...)
	ex = mapping(x)
end

init_vamp_mean(K::Int, X::AbstractArray, s=1f0) = VAMP(K, s*randn(eltype(X), size(X)[1:end-1]..., K) .+ mean(X, dims=ndims(X)))
init_vamp_sample(K::Int, X::AbstractArray) = VAMP(K,  X[repeat([:], ndims(X)-1)..., rand(1:size(X)[end], K)])
