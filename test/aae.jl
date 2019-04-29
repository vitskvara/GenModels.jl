using GenerativeModels
using Flux
using ValueHistories
using Test
using Random
include(joinpath(dirname(pathof(GenerativeModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

@testset "AAE" begin
	println("           adversarial autoencoder")

	x = GenerativeModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	Random.seed!(12345)
	model = GenerativeModels.AAE([xdim,2,ldim], [ldim,2,xdim], [ldim,3,1])
	_x = model(x)
	z = model.pz(N)

	# alternative constructor
	model = GenerativeModels.AAE(xdim, ldim, 3, 2)
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model)) 
	@test length(model.encoder.layers) == 3
	@test length(model.decoder.layers) == 3
	@test length(model.discriminator.layers) == 2

	# constructor with specified hdim
	model = GenerativeModels.AAE(xdim, ldim, 3, 2, hdim=10)
	# for training check
	frozen_params = getparams(model)
	@test length(model.encoder.layers) == 3
	@test length(model.decoder.layers) == 3
	@test length(model.discriminator.layers) == 2
	@test size(model.encoder.layers[1].W,1) == 10
	@test size(model.encoder.layers[end].W,2) == 10
	@test size(model.discriminator.layers[1].W,1) == 10

	# loss functions
	ael = GenerativeModels.aeloss(model,x)
	dl = GenerativeModels.dloss(model,x)
	dl = GenerativeModels.dloss(model,x,z)
	gl = GenerativeModels.gloss(model,x)
	@test typeof(ael) <: Flux.Tracker.TrackedReal	
	@test typeof(dl) <: Flux.Tracker.TrackedReal
	@test typeof(gl) <: Flux.Tracker.TrackedReal
	ael, dl, gl = GenerativeModels.getlosses(model,x)
	aelz, dlz, glz = GenerativeModels.getlosses(model,x,z)
	@test (ael, gl) == (aelz, glz)

	# tracking
	hist = MVHistory()
	GenerativeModels.track!(model, hist, x)
	GenerativeModels.track!(model, hist, x)
	is, ls = get(hist, :aeloss)
	@test ls[1] == ael
	@test ls[1] == ls[2]
	is, gls = get(hist, :gloss)
	@test gls[1] == gl
	@test gls[1] == gls[2]

	# test of basic training
	# test proper updating of autoencoder
	eparams = getparams(model.encoder)
	decparams = getparams(model.decoder)
	disparams  = getparams(model.discriminator)
	aeopt = ADAM()
	ael = GenerativeModels.aeloss(model,x)
	Flux.Tracker.back!(ael)
	GenerativeModels.update!((model.encoder,model.decoder), aeopt)
	@test all(paramchange(eparams, model.encoder))
	@test all(paramchange(decparams, model.decoder))
	@test !any(paramchange(disparams, model.discriminator))
	@test all(model.encoder.layers[end].W.data .== model.f_encoder.layers[end].W)

	# test proper updating of discriminator
	eparams = getparams(model.encoder)
	decparams = getparams(model.decoder)
	disparams  = getparams(model.discriminator)
	dopt = ADAM()
	dl = GenerativeModels.dloss(model,x)
	Flux.Tracker.back!(dl)
	@test all(model.encoder.layers[1].W.grad .== 0) 
	GenerativeModels.update!(model.discriminator, dopt)
	@test !any(paramchange(eparams, model.encoder))
	@test !any(paramchange(decparams, model.decoder))
	@test all(paramchange(disparams, model.discriminator))
	@test all(model.discriminator.layers[end].W.data .== model.f_discriminator.layers[end].W)

	# test proper updating of encoder with gloss
	eparams = getparams(model.encoder)
	decparams = getparams(model.decoder)
	disparams = getparams(model.discriminator)
	gopt = ADAM()
	gl = GenerativeModels.gloss(model,x)
	Flux.Tracker.back!(gl)
	@test all(model.discriminator.layers[1].W.grad .== 0) 
	GenerativeModels.update!(model.encoder, gopt)
	@test all(paramchange(eparams, model.encoder))
	@test !any(paramchange(decparams, model.decoder))
	@test !any(paramchange(disparams, model.discriminator))

	# test the fit function
	hist = MVHistory()
	GenerativeModels.fit!(model, x, 5, 1000, cbit=5, history = hist, verb = false)
	is, ls = get(hist, :aeloss)
	@test ls[1] > ls[end] 

end