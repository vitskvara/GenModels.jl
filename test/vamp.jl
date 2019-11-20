using GenModels
using Flux
using ValueHistories
using Test
using Random
include(joinpath(dirname(pathof(GenModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

Random.seed!(12345)

@testset "VAMP" begin
	println("           Vamp prior")

	function test_vamp(pz)
		ps = params(pz)
		@test length(ps) == 1
		@test size(collect(ps)[1]) == (xdim, K)
		@test pz.K == K

		sx = GenModels.sampleVamp(pz,5)
		@test size(sx) == (xdim,5) # test of size
		sx = GenModels.sampleVamp(pz,5,1)
		@test size(sx) == (xdim,5) # test of size
		@test all(map(i -> sx[:,i] == sx[:,i+1], 1:4)) # test that all the columns are the same
		try # test that querying a pseudoinput k > K results in an error
			sx = GenModels.sampleVamp(pz,5,5)
		catch e
			@test isa(e, ErrorException)
		end
		
		l = Dense(xdim, ldim)
		z = GenModels.encodeSampleVamp(pz, l, 5)
		@test size(z) == (ldim, 5) # test size
		z = GenModels.encodeSampleVamp(pz, l, 5, 1)
		@test size(z) == (ldim, 5)
		@test all(map(i -> z[:,i] == z[:,i+1], 1:4))	
	end

	K = 4
	pz = VAMP(K, xdim)
	test_vamp(pz)

	# now test it with a waae net 
	data = randn(Float32,4,2,1,8);
	m,n,c,k = size(data)
	# now setup the convolutional net
	insize = (m,n,c)
	latentdim = 2
	nconv = 2
	kernelsize = 3
	channels = (2,4)
	scaling = [(2,2),(1,1)]
	disc_nlayers = 2
	pz = VAMP(K, insize)
	model = GenModels.ConvWAAE(insize, latentdim, disc_nlayers, nconv, kernelsize, 
		channels, scaling, pz; kernel = GenModels.imq)
	model.pz(N::Int) = GenModels.encodeSampleVamp(model.pz, model.encoder, N)
	frozen_params_pz = getparams(model.pz)
	frozen_params = getparams(model)

	Y = model(data)
	@test size(Y) == size(data)
	# test training
	hist = MVHistory()
	GenModels.fit!(model, data, 4, 50, cbit=1, history=hist, verb=false)
	@test all(paramchange(frozen_params, model))	
	@test all(paramchange(frozen_params_pz, model.pz))	
	(i,ls) = get(hist,:aeloss)
	# @test ls[end] < ls[1]
	# encoding
	@test size(GenModels.encode(model, data)) == (latentdim,k)
	@test size(GenModels.encode(model, data,3)) == (latentdim,k)

	# test the initializers
	X = randn(3,4,10) .+ 5.0
	v = GenModels.init_vamp_mean(K, X)
	@test size(v.pseudoinputs) == (3,4,K)
	v = GenModels.init_vamp_sample(K, X)
	@test size(v.pseudoinputs) == (3,4,K)
end

Random.seed!()