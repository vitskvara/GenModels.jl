using GenModels
using Test
using ValueHistories
using Flux
using Random
include(joinpath(dirname(pathof(GenModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

@testset "VAE" begin
    println("           variational autoencoder")

	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
    Random.seed!(12345)

	# this has unit variance on output
    model = GenModels.VAE([xdim,2,2*ldim], [ldim,2,xdim])
    @test !GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == ldim
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
	@test typeof(_x) <: Flux.TrackedArray{GenModels.Float,2}
	# encoding
	@test size(GenModels.encode(model, x)) == (ldim,N)
	@test size(GenModels.encode(model, x, 3)) == (ldim,N)

	# loss functions
	kl = GenModels.KL(model,x)
	@test typeof(kl) == Flux.Tracker.TrackedReal{GenModels.Float}
	ll = GenModels.loglikelihood(model,x)
	@test typeof(ll) == Flux.Tracker.TrackedReal{GenModels.Float}
	llm = GenModels.loglikelihood(model,x,10)
	@test typeof(llm) == Flux.Tracker.TrackedReal{GenModels.Float}
	l = GenModels.loss(model, x, 10, 0.01)
	@test typeof(l) == Flux.Tracker.TrackedReal{GenModels.Float}
	ls = GenModels.getlosses(model, x, 10, 0.01)
	@test sum(map(x->abs(x[1]-x[2]), zip(ls, (kl-llm, -llm, kl))))/3 < 2e-1
	# its never the same because of the middle stochastic layer
	# tracking
	hist = MVHistory()
	GenModels.track!(model, hist, x, 10, 0.01)
	GenModels.track!(model, hist, x, 10, 0.01)
	is, ls = get(hist, :loss)
	@test abs(ls[1] - l) < 2e-1
	@test abs(ls[1] - ls[2]) < 2e-1
	# training
	GenModels.fit!(model, x, 5, 100, beta =0.1, cbit=5, history = hist, verb = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model))	
	# sample
	gx = GenModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,1}
	gx = GenModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,2}
	@test size(gx) == (xdim,5)

	###########################################################
    ### VAE with estimated diagonal of covariance on output ###
    ###########################################################
    model = GenModels.VAE([xdim,2,2*ldim], [ldim,2,xdim*2], variant = :diag)
	@test !GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == ldim
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (2*xdim,N)
	# loss functions
	prels = GenModels.getlosses(model, x, 10, 0.01)
	GenModels.fit!(model, x, 5, 500, beta =0.1, runtype = "fast")
	postls = GenModels.getlosses(model, x, 10, 0.01)
	@test any(x->x[1]>x[2], zip(prels, postls))
	gx = GenModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,1}
	gx = GenModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,2}
	@test size(gx) == (xdim,5)

	##########################################
    ### VAE with scalar variance on output ###
    ##########################################
    model = GenModels.VAE([xdim,2,2*ldim], [ldim,2,xdim + 1], variant = :scalar)
	@test !GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == ldim
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (1+xdim,N)
	# loss functions
	prels = GenModels.getlosses(model, x, 10, 0.01)
	GenModels.fit!(model, x, 5, 100, beta =0.1, runtype = "fast")
	postls = GenModels.getlosses(model, x, 10, 0.01)
	@test all(x->x[1]>x[2], zip(prels, postls))
	gx = GenModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,1}
	gx = GenModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,2}
	@test size(gx) == (xdim,5)

	# alternative constructor test
	model = GenModels.VAE(xdim, ldim, 4, hdim=10)
	@test !GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == ldim
	@test length(model.encoder.layers) == 4
	@test length(model.decoder.layers) == 4
	@test size(model.encoder(x)) == (ldim*2, N)
	@test size(model(x)) == (xdim, N)
	@test size(model.encoder.layers[1].W,1) == 10
	@test size(model.encoder.layers[end].W,2) == 10

	model = GenModels.VAE(xdim, ldim, 4, variant = :scalar)
	@test !GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == ldim
	@test size(model(x)) == (xdim+1, N)

	model = GenModels.VAE(xdim, ldim, 4, variant = :diag)
	@test !GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == ldim
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
	model = GenModels.ConvVAE(insize, latentdim, nconv, kernelsize, channels, scaling;
		batchnorm = batchnorm)
	@test GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == latentdim
	hist = MVHistory()
	frozen_params = getparams(model)
	@test size(model(data)) == size(data)
	@test size(model.encoder(data)) == (latentdim*2,k)
	GenModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false);
	@test all(paramchange(frozen_params, model))	
	(i,ls) = get(hist,:loss)
	@test ls[end] < ls[1]
	gx = GenModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,4}
	gx = GenModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,4}
	@test size(gx) == (m,n,c,5)

	# diag
	model = GenModels.ConvVAE(insize, latentdim, nconv, kernelsize, channels, scaling; 
		variant=:diag, batchnorm = batchnorm)
	@test GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == latentdim
	hist = MVHistory()
	frozen_params = getparams(model)
	@test size(model(data)) == (m,n,c*2,k)
	@test size(model.encoder(data)) == (latentdim*2,k)
	GenModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false);
	@test all(paramchange(frozen_params, model))	
	(i,ls) = get(hist,:loss)
	@test ls[end] < ls[1]
	gx = GenModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,4}
	gx = GenModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,4}
	@test size(gx) == (m,n,c,5)

	#scalar
	model = GenModels.ConvVAE(insize, latentdim, nconv, kernelsize, channels, scaling; 
		variant=:scalar, batchnorm = batchnorm)
	@test GenModels.isconvvae(model)
    @test GenModels.getlsize(model) == latentdim
	hist = MVHistory()
	frozen_params = getparams(model)
	@test size(model(data)) == (m,n,c*2,k)
	@test size(model.encoder(data)) == (latentdim*2,k)
	GenModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false);
	@test all(paramchange(frozen_params, model))	
	(i,ls) = get(hist,:loss)
	@test ls[end] < ls[1]
	gx = GenModels.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,4}
	gx = GenModels.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{GenModels.Float,4}
	@test size(gx) == (m,n,c,5)
	# encoding
	@test size(GenModels.encode(model, data)) == (latentdim,k)
	@test size(GenModels.encode(model, data,3)) == (latentdim,k)

end