using GenModels
using Flux
using ValueHistories
using Test
using Random
include(joinpath(dirname(pathof(GenModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

@testset "WAAE" begin
	println("           wasserstein-adversarial autoencoder")

	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	Random.seed!(12345)
	model = GenModels.WAAE([xdim,2,ldim], [ldim,2,xdim], [ldim,4,1])
	_x = model(x)
	z = model.pz(N)
	@test size(_x) == size(x)
	@test size(z,2) == N

	# alternative constructor
	nlayers = 3
	disc_nlayers = 2
	model = GenModels.WAAE(xdim, ldim, nlayers, disc_nlayers; 
		hdim=10, kernel = GenModels.imq)
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model)) 
	@test length(model.encoder.layers) == nlayers
	@test length(model.decoder.layers) == nlayers
	@test length(model.discriminator.layers) ==disc_nlayers
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
	s = 1.0
	λ = 1.0
	γ = 1.0
	l = GenModels.loss(model,x,s,λ,γ)
	ael = GenModels.aeloss(model,x)
	mmd = GenModels.MMD(model,x,s)
	dl = GenModels.dloss(model,x)
	dl = GenModels.dloss(model,x,z)
	gl = GenModels.gloss(model,x)
	@test typeof(Flux.Tracker.data(l)) == GenModels.Float 
	@test typeof(l) <: Flux.Tracker.TrackedReal
	@test typeof(ael) <: Flux.Tracker.TrackedReal
	@test typeof(mmd) <: Flux.Tracker.TrackedReal
	@test typeof(dl) <: Flux.Tracker.TrackedReal
	@test typeof(gl) <: Flux.Tracker.TrackedReal
	l, ael, dl, gl, mmd = GenModels.getlosses(model,x,s,λ,γ)
	lz, aelz, dlz, glz, mmdz = GenModels.getlosses(model,x,z,s,λ,γ)
	@test l != lz
	@test ael == aelz
	@test dl != dlz
	@test gl == glz
	@test mmd != mmdz

	# tracking
	hist = MVHistory()
	GenModels.track!(model, hist, x, s, λ, γ)
	GenModels.track!(model, hist, x, s, λ, γ)
	is, ls = get(hist, :loss)
	@test ls[1] != l
	@test ls[1] != ls[2]
	is, aels = get(hist, :aeloss)
	@test aels[1] == ael
	@test aels[1] == aels[2]
	is, dls = get(hist, :dloss)
	@test dls[1] != dl
	@test dls[1] != dls[2]
	is, gls = get(hist, :gloss)
	@test gls[1] == gl
	@test gls[1] == gls[2]
	is, mmds = get(hist, :mmd)
	@test mmds[1] != mmd
	@test mmds[1] != mmds[2]

	# test of basic training
	# test proper updating of autoencoder
	eparams = getparams(model.encoder)
	dparams = getparams(model.decoder)
	discparams = getparams(model.discriminator)
	opt = ADAM()
	l = GenModels.MMD(model,x,s)
	Flux.Tracker.back!(l)
	GenModels.update!(model.encoder, opt)
	@test all(paramchange(eparams, model.encoder))
	@test !any(paramchange(dparams, model.decoder))
	@test !any(paramchange(discparams, model.discriminator))
	# reconstruction loss
	eparams = getparams(model.encoder)
	opt = ADAM()
	l = GenModels.aeloss(model,x)
	Flux.Tracker.back!(l)
	GenModels.update!(model, opt)
	@test all(paramchange(eparams, model.encoder))
	@test all(paramchange(dparams, model.decoder))
	@test !any(paramchange(discparams, model.discriminator))
	# discriminator loss
	eparams = getparams(model.encoder)
	dparams = getparams(model.decoder)
	discparams = getparams(model.discriminator)
	opt = ADAM()
	l = GenModels.dloss(model,x)
	Flux.Tracker.back!(l)
	GenModels.update!(model, opt)
	@test !any(paramchange(eparams, model.encoder))
	@test !any(paramchange(dparams, model.decoder))
	@test all(paramchange(discparams, model.discriminator))
	# generator loss
	eparams = getparams(model.encoder)
	dparams = getparams(model.decoder)
	discparams = getparams(model.discriminator)
	opt = ADAM()
	l = GenModels.gloss(model,x)
	Flux.Tracker.back!(l)
	GenModels.update!(model, opt)
	@test all(paramchange(eparams, model.encoder))
	@test !any(paramchange(dparams, model.decoder))
	@test !any(paramchange(discparams, model.discriminator))
	# total loss
	eparams = getparams(model.encoder)
	dparams = getparams(model.decoder)
	discparams = getparams(model.discriminator)
	opt = ADAM()
	l = GenModels.loss(model,x,s,λ,γ)
	Flux.Tracker.back!(l)
	GenModels.update!(model, opt)
	@test all(paramchange(eparams, model.encoder))
	@test all(paramchange(dparams, model.decoder))
	@test all(paramchange(discparams, model.discriminator))

	# test sampling
	@test size(GenModels.sample(model)) == (xdim,1)
	@test size(GenModels.sample(model,7)) == (xdim,7)
	
	# test the fit function
	hist = MVHistory()
	GenModels.fit!(model, x, 5, 1000; γ=γ, λ=λ, σ=s, cbit=5, history = hist, verb = false);
	is, ls = get(hist, :aeloss)
	@test ls[1] > ls[end] 

	# convolutional WAAE
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
	disc_nlayers = 3
	model = GenModels.ConvWAAE(insize, latentdim, disc_nlayers, nconv, kernelsize, 
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
