using Flux
using ValueHistories
using CuArrays
using Test
using Random
using GenerativeModels
include(joinpath(dirname(pathof(GenerativeModels)), "../test/test_utils.jl"))

xdim = 5
ldim = 1
N = 10

@testset "flux utils" begin
	# iscuarray
	x = randn(4,10)
	@test !GenerativeModels.iscuarray(x)
	x = x |> gpu
	@test GenerativeModels.iscuarray(x)
	model = Flux.Chain(Flux.Dense(4, ldim), Flux.Dense(ldim, 4)) |> gpu
	_x = model(x)
	@test GenerativeModels.iscuarray(_x)
	X = randn(4,4,1,1) |> gpu
	@test GenerativeModels.iscuarray(X)
end

@testset "model utils" begin
	x = fill(0.0,5,5) |> gpu
	sd = fill(1.0,5) |> gpu
	# the version where sigma is a vector (scalar variance)
	@test sim(GenerativeModels.loglikelihood(x,x,sd), 0.0 - GenerativeModels.l2pi/2*5)
	@test sim(GenerativeModels.loglikelihood(x,x,sd), 0.0 - GenerativeModels.l2pi/2*5)
end

@testset "AE-GPU" begin
	x = GenerativeModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	Random.seed!(12345)
	model = GenerativeModels.AE([xdim,2,ldim], [ldim,2,xdim]) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model))

	@test typeof(gx) == CuArray{GenerativeModels.Float,2}
	@test typeof(_x) <: TrackedArray{GenerativeModels.Float,2}    
	hist = MVHistory()
	GenerativeModels.fit!(model, x, 5, 1000, cbit=100, history = hist, verb=false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	@test ls[end] < 1e-4
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "VAE-GPU" begin
	x = GenerativeModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	# unit VAE
	Random.seed!(12345)
    model = GenerativeModels.VAE([xdim,2,2*ldim], [ldim,2,xdim]) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)
	@test !all(paramchange(frozen_params, model)) 
	@test typeof(gx) == CuArray{GenerativeModels.Float,2}
	@test typeof(_x) <: TrackedArray{GenerativeModels.Float,2}    
	hist = MVHistory()
	GenerativeModels.fit!(model, x, 10, 50, beta =0.1, cbit=5, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 

	# diag VAE
	Random.seed!(12345)
    model = GenerativeModels.VAE([xdim,2,2*ldim], [ldim,2,xdim*2], variant = :diag) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)
	@test typeof(gx) == CuArray{GenerativeModels.Float,2}
	@test typeof(_x) <: TrackedArray{GenerativeModels.Float,2}    
	hist = MVHistory()
	GenerativeModels.fit!(model, x, 5, 50, beta =0.1, cbit=5, history = hist, verb = false,
		usegpu = true)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "ConvVAE - GPU" begin
	m,n,c,k = (8,8,1,N)	
    Random.seed!(12345)
    X = randn(GenerativeModels.Float, m,n,c,k)
    gX = X |> gpu
    nconv = 2
    kernelsize = 3
    channels = (2,4)
    scaling = 2
    # unit model
    model = GenerativeModels.ConvVAE((m,n,c), ldim, nconv, kernelsize, channels, scaling) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenerativeModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenerativeModels.Float, 4}
	@test GenerativeModels.iscuarray(_X)
	hist = MVHistory()
	GenerativeModels.fit!(model, X, 5, 10, beta =0.01, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 

    # scalar model
    model = GenerativeModels.ConvVAE((m,n,c), ldim, nconv, kernelsize, channels, scaling; 
    	variant = :scalar) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenerativeModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenerativeModels.Float, 4}
	@test GenerativeModels.iscuarray(_X)
	hist = MVHistory()
	GenerativeModels.fit!(model, X, 5, 10, beta =0.01, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 

    # diag model
    model = GenerativeModels.ConvVAE((m,n,c), ldim, nconv, kernelsize, channels, scaling; 
    	variant = :diag) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenerativeModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenerativeModels.Float, 4}
	@test GenerativeModels.iscuarray(_X)
	hist = MVHistory()
	GenerativeModels.fit!(model, X, 5, 10, beta =0.01, history = hist, verb = false,
		usegpu = true, memoryefficient = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model))

end

@testset "TSVAE-GPU" begin
	x = GenerativeModels.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	gx = x |> gpu
	Random.seed!(12345)
    model = GenerativeModels.TSVAE(xdim, ldim, 2) |> gpu
	_x = model(gx)
	# for training check
	frozen_params = getparams(model)

	@test typeof(gx) == CuArray{GenerativeModels.Float,2}
	@test typeof(_x) <: TrackedArray{GenerativeModels.Float,2}    
	history = (MVHistory(),MVHistory())
    GenerativeModels.fit!(model, x, 5, 500; history = history, verb = false, usegpu = true,
    	memoryefficient = false)
    _,ls = get(history[1],:loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model)) 
end

@testset "ConvTSVAE-GPU" begin
	m,n,c,k = (8,8,1,N)	
    Random.seed!(12345)
    X = randn(GenerativeModels.Float, m,n,c,k) |> gpu
    gX = X |> gpu
    nlayers = 2
    kernelsize = 3
    channels = (2,4)
    scaling = 2
    # unit model
    model = GenerativeModels.ConvTSVAE((m,n,c), ldim, nlayers, kernelsize, channels, scaling) |> gpu
    _X = model(gX)
	# for training check
	frozen_params = getparams(model)
	@test typeof(_X) <: TrackedArray{GenerativeModels.Float,4}    
	@test typeof(_X.data) == CuArray{GenerativeModels.Float, 4}
	@test GenerativeModels.iscuarray(_X)
	hist = (MVHistory(), MVHistory())
	GenerativeModels.fit!(model, X, 5, 40, beta = 1.0, history = hist, verb = false,
		usegpu = true, memoryefficient = false, cbit=1, η = 0.1);
	for h in hist
		is, ls = get(h, :loss)
		@test ls[1] > ls[end] 
	end
	# were the layers realy trained?
	@test all(paramchange(frozen_params, model))
end