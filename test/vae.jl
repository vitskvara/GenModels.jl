using GenerativeModels
using Test
using ValueHistories
using Flux
using Random
include(joinpath(dirname(pathof(GenerativeModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

@testset "VAE" begin
    println("           variational autoencoder")

	x = GenerativeModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
    Random.seed!(12345)

	# this has unit variance on output
    model = GenerativeModels.VAE([xdim,2,2*ldim], [ldim,2,xdim])
    @test !GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == ldim
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model)) 
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (xdim,N)
	# test output types
	@test typeof(_x) <: Flux.TrackedArray{GenerativeModels.Float,2}

	# loss functions
	kl = GenerativeModels.KL(model,x)
	@test typeof(kl) == Flux.Tracker.TrackedReal{GenerativeModels.Float}
	ll = GenerativeModels.loglikelihood(model,x)
	@test typeof(ll) == Flux.Tracker.TrackedReal{GenerativeModels.Float}
	llm = GenerativeModels.loglikelihood(model,x,10)
	@test typeof(llm) == Flux.Tracker.TrackedReal{GenerativeModels.Float}
	l = GenerativeModels.loss(model, x, 10, 0.01)
	@test typeof(l) == Flux.Tracker.TrackedReal{GenerativeModels.Float}
	ls = GenerativeModels.getlosses(model, x, 10, 0.01)
	@test sum(map(x->abs(x[1]-x[2]), zip(ls, (kl-llm, -llm, kl))))/3 < 2e-1
	# its never the same because of the middle stochastic layer
	# tracking
	hist = MVHistory()
	GenerativeModels.track!(model, hist, x, 10, 0.01)
	GenerativeModels.track!(model, hist, x, 10, 0.01)
	is, ls = get(hist, :loss)
	@test abs(ls[1] - l) < 2e-1
	@test abs(ls[1] - ls[2]) < 1e-1
	# training
	GenerativeModels.fit!(model, x, 5, 100, beta =0.1, cbit=5, history = hist, verb = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model))	
	# sample
	gx = GenerativeModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,1}
	gx = GenerativeModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,2}
	@test size(gx) == (xdim,5)

	###########################################################
    ### VAE with estimated diagonal of covariance on output ###
    ###########################################################
    model = GenerativeModels.VAE([xdim,2,2*ldim], [ldim,2,xdim*2], variant = :diag)
	@test !GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == ldim
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (2*xdim,N)
	# loss functions
	prels = GenerativeModels.getlosses(model, x, 10, 0.01)
	GenerativeModels.fit!(model, x, 5, 500, beta =0.1, runtype = "fast")
	postls = GenerativeModels.getlosses(model, x, 10, 0.01)
	@test any(x->x[1]>x[2], zip(prels, postls))
	gx = GenerativeModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,1}
	gx = GenerativeModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,2}
	@test size(gx) == (xdim,5)

	##########################################
    ### VAE with scalar variance on output ###
    ##########################################
    model = GenerativeModels.VAE([xdim,2,2*ldim], [ldim,2,xdim + 1], variant = :scalar)
	@test !GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == ldim
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (1+xdim,N)
	# loss functions
	prels = GenerativeModels.getlosses(model, x, 10, 0.01)
	GenerativeModels.fit!(model, x, 5, 100, beta =0.1, runtype = "fast")
	postls = GenerativeModels.getlosses(model, x, 10, 0.01)
	@test all(x->x[1]>x[2], zip(prels, postls))
	gx = GenerativeModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,1}
	gx = GenerativeModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,2}
	@test size(gx) == (xdim,5)

	# alternative constructor test
	model = GenerativeModels.VAE(xdim, ldim, 4, hdim=10)
	@test !GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == ldim
	@test length(model.encoder.layers) == 4
	@test length(model.decoder.layers) == 4
	@test size(model.encoder(x)) == (ldim*2, N)
	@test size(model(x)) == (xdim, N)
	@test size(model.encoder.layers[1].W,1) == 10
	@test size(model.encoder.layers[end].W,2) == 10

	model = GenerativeModels.VAE(xdim, ldim, 4, variant = :scalar)
	@test !GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == ldim
	@test size(model(x)) == (xdim+1, N)

	model = GenerativeModels.VAE(xdim, ldim, 4, variant = :diag)
	@test !GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == ldim
	@test size(model(x)) == (xdim*2, N)

	# convolutional VAES
	data = randn(Float32,32,16,1,8);
	m,n,c,k = size(data)
	insize = (m,n,c)
	latentdim = 2
	nconv = 3
	kernelsize = 3
	channels = (2,4,6)
	scaling = [(2,2),(2,2),(1,1)]
	batchnorm = true
	# unit
	model = GenerativeModels.ConvVAE(insize, latentdim, nconv, kernelsize, channels, scaling;
		batchnorm = batchnorm)
	@test GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == latentdim
	hist = MVHistory()
	frozen_params = getparams(model)
	@test size(model(data)) == size(data)
	@test size(model.encoder(data)) == (latentdim*2,k)
	GenerativeModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false);
	@test all(paramchange(frozen_params, model))	
	(i,ls) = get(hist,:loss)
	@test ls[end] < ls[1]
	gx = GenerativeModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,4}
	gx = GenerativeModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,4}
	@test size(gx) == (m,n,c,5)

	# diag
	model = GenerativeModels.ConvVAE(insize, latentdim, nconv, kernelsize, channels, scaling; 
		variant=:diag, batchnorm = batchnorm)
	@test GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == latentdim
	hist = MVHistory()
	frozen_params = getparams(model)
	@test size(model(data)) == (m,n,c*2,k)
	@test size(model.encoder(data)) == (latentdim*2,k)
	GenerativeModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false);
	@test all(paramchange(frozen_params, model))	
	(i,ls) = get(hist,:loss)
	@test ls[end] < ls[1]
	gx = GenerativeModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,4}
	gx = GenerativeModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,4}
	@test size(gx) == (m,n,c,5)

	#scalar
	model = GenerativeModels.ConvVAE(insize, latentdim, nconv, kernelsize, channels, scaling; 
		variant=:scalar, batchnorm = batchnorm)
	@test GenerativeModels.isconvvae(model)
    @test GenerativeModels.getlsize(model) == latentdim
	hist = MVHistory()
	frozen_params = getparams(model)
	@test size(model(data)) == (m,n,c*2,k)
	@test size(model.encoder(data)) == (latentdim*2,k)
	GenerativeModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false);
	@test all(paramchange(frozen_params, model))	
	(i,ls) = get(hist,:loss)
	@test ls[end] < ls[1]
	gx = GenerativeModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,4}
	gx = GenerativeModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenerativeModels.Float,4}
	@test size(gx) == (m,n,c,5)
end