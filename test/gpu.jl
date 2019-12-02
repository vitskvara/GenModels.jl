using Flux
using ValueHistories
using CuArrays
using Test
using Random
using GenModels
include(joinpath(dirname(pathof(GenModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

@testset "flux utils" begin
	# iscuarray
	x = randn(4,10)
	@test !GenModels.iscuarray(x)
	x = x |> gpu
	@test GenModels.iscuarray(x)
	model = Flux.Chain(Flux.Dense(4, ldim), Flux.Dense(ldim, 4)) |> gpu
	_x = model(x)
	@test GenModels.iscuarray(_x)
	X = randn(4,4,1,1) |> gpu
	@test GenModels.iscuarray(X)
end

@testset "model utils" begin
	x = fill(0.0,5,5) |> gpu
	sd = fill(1.0,5) |> gpu
	# the version where sigma is a vector (scalar variance)
	@test sim(GenModels.loglikelihood(x,x,sd), 0.0 - GenModels.l2pi/2*5)
	@test sim(GenModels.loglikelihood(x,x,sd), 0.0 - GenModels.l2pi/2*5)
end

@testset "AE-GPU" begin
	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	Random.seed!(12345)
	model = GenModels.AE([xdim,2,ldim], [ldim,2,xdim]) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model))

	@test typeof(gx) == CuArray{GenModels.Float,2, Nothing}
	@test typeof(_x.data) == CuArray{GenModels.Float,2, Nothing}
	@test typeof(_x) <: TrackedArray{GenModels.Float,2}    
	hist = MVHistory()
	GenModels.fit!(model, x, 5, 1000, cbit=100, history = hist, verb=false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	@test ls[end] < 1e-4
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "VAE-GPU" begin
	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	# unit VAE
	Random.seed!(12345)
    model = GenModels.VAE([xdim,2,2*ldim], [ldim,2,xdim]) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model)) 

	@test typeof(_x.data) == CuArray{GenModels.Float,2, Nothing}
	@test typeof(_x) <: TrackedArray{GenModels.Float,2}    
	hist = MVHistory()
	GenModels.fit!(model, x, 10, 50, beta =0.1, cbit=5, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 

	# diag VAE
	Random.seed!(12345)
    model = GenModels.VAE([xdim,2,2*ldim], [ldim,2,xdim*2], variant = :diag) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)

	@test typeof(_x.data) == CuArray{GenModels.Float,2, Nothing}
	@test typeof(_x) <: TrackedArray{GenModels.Float,2}    
	hist = MVHistory()
	GenModels.fit!(model, x, 5, 50, beta =0.1, cbit=5, history = hist, verb = false,
		usegpu = true)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "ConvVAE - GPU" begin
	m,n,c,k = (8,8,1,N)	
    Random.seed!(12345)
    X = randn(GenModels.Float, m,n,c,k)
    gX = X |> gpu
    nconv = 2
    kernelsize = 3
    channels = (2,4)
    scaling = 2
    # unit model
    model = GenModels.ConvVAE((m,n,c), ldim, nconv, kernelsize, channels, scaling) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenModels.Float, 4, Nothing}
	@test GenModels.iscuarray(_X)
	hist = MVHistory()
	GenModels.fit!(model, X, 5, 10, beta =0.01, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 

    # scalar model
    model = GenModels.ConvVAE((m,n,c), ldim, nconv, kernelsize, channels, scaling; 
    	variant = :scalar) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenModels.Float, 4, Nothing}
	@test GenModels.iscuarray(_X)
	hist = MVHistory()
	GenModels.fit!(model, X, 5, 10, beta =0.01, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 

    # diag model
    model = GenModels.ConvVAE((m,n,c), ldim, nconv, kernelsize, channels, scaling; 
    	variant = :diag) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenModels.Float, 4, Nothing}
	@test GenModels.iscuarray(_X)
	hist = MVHistory()
	GenModels.fit!(model, X, 5, 10, beta =0.01, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model))

end

@testset "TSVAE-GPU" begin
	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	Random.seed!(12345)
    model = GenModels.TSVAE(xdim, ldim, 2) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)

	@test typeof(_x.data) == CuArray{GenModels.Float,2, Nothing}
	@test typeof(_x) <: TrackedArray{GenModels.Float,2}    
	history = (MVHistory(),MVHistory())
    GenModels.fit!(model, x, 5, 500; history = history, verb = false, usegpu = true,
    	memoryefficient = false)
    _,ls = get(history[1],:loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "ConvTSVAE-GPU" begin
	m,n,c,k = (8,8,1,N)	
    Random.seed!(12345)
    X = randn(GenModels.Float, m,n,c,k) |> gpu
    gX = X |> gpu
    nlayers = 2
    kernelsize = 3
    channels = (2,4)
    scaling = 2
    # unit model
    model = GenModels.ConvTSVAE((m,n,c), ldim, nlayers, kernelsize, channels, scaling) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenModels.Float, 4, Nothing}
	@test GenModels.iscuarray(_X)
	hist = (MVHistory(), MVHistory())
	GenModels.fit!(model, X, 5, 40, beta = 1.0, history = hist, verb = false,
		usegpu = true, memoryefficient = false, cbit=1, η = 0.1);
	for h in hist
		is, ls = get(h, :loss)
		@test ls[1] > ls[end] 
	end
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model))
end

@testset "AAE-GPU" begin
	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	Random.seed!(12345)
	pz(n) = GenModels.randn_gpu(Float32, ldim, n)
    model = GenModels.AAE(xdim, ldim, 3, 3, pz) |> gpu
 	_x = model(gx)
	# for training check
	frozen_params = getparams(model)

	@test typeof(gx) == CuArray{GenModels.Float,2, Nothing}
	@test typeof(_x) <: TrackedArray{GenModels.Float,2}    
	history = MVHistory()
    GenModels.fit!(model, x, 5, 500; history = history, verb = false, usegpu = true,
    	memoryefficient = false)
    _,ls = get(history,:aeloss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "ConvAAE - GPU" begin
	m,n,c,k = (8,8,1,N)	
    Random.seed!(12345)
    X = randn(GenModels.Float, m,n,c,k)
    gX = X |> gpu
    nconv = 2
    kernelsize = 3
    channels = (2,4)
    scaling = 2
	pz(n) = GenModels.randn_gpu(Float32, ldim, n)
    # unit model
    model = GenModels.ConvAAE((m,n,c), ldim, 4, nconv, kernelsize, channels, scaling, 
    	pz; hdim = 10) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenModels.Float, 4, Nothing}
	@test GenModels.iscuarray(_X)
	hist = MVHistory()
	GenModels.fit!(model, X, 5, 10, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :aeloss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "WAE-GPU" begin
	x = GenModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	Random.seed!(12345)
	pz(n) = GenModels.randn_gpu(Float32, ldim, n)
    model = GenModels.WAE(xdim, ldim, 3, pz) |> gpu
 	_x = model(gx)
	# for training check
	frozen_params = getparams(model)

	@test typeof(gx) == CuArray{GenModels.Float,2, Nothing}
	@test typeof(_x) <: TrackedArray{GenModels.Float,2}    
	history = MVHistory()
    GenModels.fit!(model, x, 5, 500; σ=1.0, λ=1.0, history = history, verb = false, usegpu = true,
    	memoryefficient = false)
    _,ls = get(history,:loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "ConvWAE - GPU" begin
	m,n,c,k = (8,8,1,N)	
    Random.seed!(12345)
    X = randn(GenModels.Float, m,n,c,k)
    gX = X |> gpu
    nconv = 2
    kernelsize = 3
    channels = (2,4)
    scaling = 2
	pz(n) = GenModels.randn_gpu(Float32, ldim, n)
    # unit model
    model = GenModels.ConvWAE((m,n,c), ldim, nconv, kernelsize, channels, scaling, 
    	pz; kernel = GenModels.imq) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenModels.Float, 4, Nothing}
	@test GenModels.iscuarray(_X)
	hist = MVHistory()
	GenModels.fit!(model, X, 5, 10; σ=1.0, λ=1.0, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "VAMP - GPU" begin
	# now test it with a waae net 
	K = 4
	data = randn(Float32,4,2,1,8) |> gpu;
	m,n,c,k = size(data)
	# now setup the convolutional net
	insize = (m,n,c)
	latentdim = 2
	nconv = 2
	kernelsize = 3
	channels = (2,4)
	scaling = [(2,2),(1,1)]
	disc_nlayers = 2
	pz = VAMP(K, insize) |> gpu
	model = GenModels.ConvWAAE(insize, latentdim, disc_nlayers, nconv, kernelsize, 
		channels, scaling, pz; kernel = GenModels.imq) |> gpu
	model.pz(N::Int) = GenModels.encodeSampleVamp(model.pz, model.encoder, N)
	frozen_params_pz = getparams(model.pz)
	frozen_params = getparams(model)

	z = model.pz(10)
	@test size(z) == (latentdim, 10)
	@test typeof(z) <: TrackedArray{GenModels.Float,2}    
	@test typeof(z.data) == CuArray{GenModels.Float, 2, Nothing}
	z = GenModels.sampleVamp(model.pz,10,2)
	@test size(z) == (m,n,c, 10)
	@test typeof(z) <: TrackedArray{GenModels.Float,4}    
	@test typeof(z.data) == CuArray{GenModels.Float, 4, Nothing}
	# test training
	hist = MVHistory()
	GenModels.fit!(model, data, 4, 10, cbit=1, history=hist, verb=false)
	@test all(paramchange(frozen_params, model))	
	@test all(paramchange(frozen_params_pz, model.pz))	
end