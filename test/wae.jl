using GenModels
using Flux
using ValueHistories
using Test
using Random
include(joinpath(dirname(pathof(GenModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

@testset "WAE" begin
	println("           wasserstein autoencoder")

	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	Random.seed!(12345)
	model = GenModels.WAE([xdim,2,ldim], [ldim,2,xdim])
	_x = model(x)
	z = model.pz(N)
	@test size(_x) == size(x)
	@test size(z,2) == N

	# alternative constructor
	model = GenModels.WAE(xdim, ldim, 3; hdim=10, kernel = GenModels.imq)
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model)) 
	@test length(model.encoder.layers) == 3
	@test length(model.decoder.layers) == 3
	@test size(model.encoder.layers[1].W,1) == 10
	@test size(model.encoder.layers[end].W,2) == 10
	_x = model(x)
	z = model.pz(N)
	@test size(_x) == size(x)
	@test size(z,2) == N
	# encoding
	@test size(GenModels.encode(model, x)) == (ldim,N)
	@test size(GenModels.encode(model, x, 3)) == (ldim,N)

	# loss functions
	σ = 1.0
	λ = 1.0
	l = GenModels.loss(model,x,σ,λ)
	ael = GenModels.aeloss(model,x)
	mmd = GenModels.MMD(model,x,σ)
	@test typeof(Flux.Tracker.data(l)) == GenModels.Float 
	@test typeof(l) <: Flux.Tracker.TrackedReal
	@test typeof(ael) <: Flux.Tracker.TrackedReal
	@test typeof(mmd) <: Flux.Tracker.TrackedReal
	l, ael, mmd = GenModels.getlosses(model,x,σ,λ)
	lz, aelz, mmdz = GenModels.getlosses(model,x,z,σ,λ)
	@test l != lz
	@test ael == aelz
	@test mmd != mmdz

	# tracking
	hist = MVHistory()
	GenModels.track!(model, hist, x, σ, λ)
	GenModels.track!(model, hist, x, σ, λ)
	is, ls = get(hist, :loss)
	@test ls[1] != l
	@test ls[1] != ls[2]
	is, aels = get(hist, :aeloss)
	@test aels[1] == ael
	@test aels[1] == aels[2]
	is, mmds = get(hist, :mmd)
	@test mmds[1] != mmd
	@test mmds[1] != mmds[2]

	# test of basic training
	# test proper updating of autoencoder
	eparams = getparams(model.encoder)
	dparams = getparams(model.decoder)
	opt = ADAM()
	l = GenModels.MMD(model,x,σ)
	Flux.Tracker.back!(l)
	GenModels.update!(model.encoder, opt)
	@test all(paramchange(eparams, model.encoder))
	@test !any(paramchange(dparams, model.decoder))
	# total loss
	eparams = getparams(model.encoder)
	opt = ADAM()
	l = GenModels.loss(model,x,σ,λ)
	Flux.Tracker.back!(l)
	GenModels.update!(model, opt)
	@test all(paramchange(eparams, model.encoder))
	@test all(paramchange(dparams, model.decoder))

	# test sampling
	@test size(GenModels.sample(model)) == (xdim,1)
	@test size(GenModels.sample(model,7)) == (xdim,7)
	
	# test the fit function
	hist = MVHistory()
	GenModels.fit!(model, x, 5, 1000; σ=σ, λ=λ, cbit=5, history = hist, verb = false);
	is, ls = get(hist, :aeloss)
	@test ls[1] > ls[end] 

	# convolutional WAE
	data = randn(Float32,32,16,1,8);
	m,n,c,k = size(data)
	# now setup the convolutional net
	insize = (m,n,c)
	latentdim = 2
	nconv = 3
	kernelsize = 3
	channels = (2,4,6)
	scaling = [(2,2),(2,2),(1,1)]
	batchnorm = true
	model = GenModels.ConvWAE(insize, latentdim, nconv, kernelsize, 
		channels, scaling; kernel = GenModels.imq, batchnorm = batchnorm)
	Y = model(data)
	@test size(Y) == size(data)
	# test training
	hist = MVHistory()
	frozen_params = getparams(model)
	@test size(model(data)) == size(data)
	@test size(model.encoder(data)) == (latentdim,k)
	GenModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false)
	@test all(paramchange(frozen_params, model))	
	(i,ls) = get(hist,:aeloss)
	@test ls[end] < ls[1]
	# encoding
	@test size(GenModels.encode(model, data)) == (latentdim,k)
	@test size(GenModels.encode(model, data,3)) == (latentdim,k)

end
