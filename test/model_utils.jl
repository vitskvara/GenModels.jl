using GenerativeModels
using Test
using Random
using StatsBase
using Flux
include(joinpath(dirname(pathof(GenerativeModels)), "../test/test_utils.jl"))

Random.seed!(12345)

@testset "model utils" begin
    # KL
    @test GenerativeModels.KL(0.0, 1.0) == 0.0
    @test GenerativeModels.KL(fill(0.0,5), fill(1.0,5)) == 0.0
    @test GenerativeModels.KL(fill(0.0,5,5), fill(1.0,5,5)) == 0.0

    # loglikelihood
    @test GenerativeModels.loglikelihood(0.0,0.0) == 0.0 - GenerativeModels.l2pi/2
    @test GenerativeModels.loglikelihood(5,5) == 0.0 - GenerativeModels.l2pi/2
    @test GenerativeModels.loglikelihood(0.0,0.0,1.0) == 0.0 - GenerativeModels.l2pi/2
	@test GenerativeModels.loglikelihood(5,5,1) == 0.0 - GenerativeModels.l2pi/2

    @test sim(GenerativeModels.loglikelihood(fill(0.0,5),fill(0.0,5)), 0.0 - GenerativeModels.l2pi/2*5)
    @test sim(GenerativeModels.loglikelihood(fill(3,5),fill(3,5)), 0.0 - GenerativeModels.l2pi/2*5)
    @test sim(GenerativeModels.loglikelihood(fill(0.0,5),fill(0.0,5),fill(1.0,5)), 0.0 - GenerativeModels.l2pi/2*5)
	@test sim(GenerativeModels.loglikelihood(fill(5,5),fill(5,5),fill(1,5)), 0.0 - GenerativeModels.l2pi/2*5)

	# maybe define a different behaviour for vectors and matrices?
	# this would then return a vector
	@test sim(GenerativeModels.loglikelihood(fill(0.0,5,5),fill(0.0,5,5)), 0.0 - GenerativeModels.l2pi/2*5)
    @test sim(GenerativeModels.loglikelihood(fill(3,5,5),fill(3,5,5)), 0.0 - GenerativeModels.l2pi/2*5)
    @test sim(GenerativeModels.loglikelihood(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5,5)), 0.0 - GenerativeModels.l2pi/2*5)
	@test sim(GenerativeModels.loglikelihood(fill(5,5,5),fill(5,5,5),fill(1,5,5)), 0.0 - GenerativeModels.l2pi/2*5)
	
	# the version where sigma is a vector (scalar variance)
	@test sim(GenerativeModels.loglikelihood(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5)), 0.0 - GenerativeModels.l2pi/2*5)
	@test sim(GenerativeModels.loglikelihood(fill(5,5,5),fill(5,5,5),fill(1,5)), 0.0 - GenerativeModels.l2pi/2*5)
	
	# loglikelihoodopt
	@test GenerativeModels.loglikelihoodopt(0.0,0.0) == 0.0
    @test GenerativeModels.loglikelihoodopt(5,5) == 0.0
    @test GenerativeModels.loglikelihoodopt(0.0,0.0,1.0) == 0.0
	@test GenerativeModels.loglikelihoodopt(5,5,1) == 0.0

    @test sim(GenerativeModels.loglikelihoodopt(fill(0.0,5),fill(0.0,5)), 0.0)
    @test sim(GenerativeModels.loglikelihoodopt(fill(3,5),fill(3,5)), 0.0)
    @test sim(GenerativeModels.loglikelihoodopt(fill(0.0,5),fill(0.0,5),fill(1.0,5)), 0.0)
	@test sim(GenerativeModels.loglikelihoodopt(fill(5,5),fill(5,5),fill(1,5)), 0.0)

	# maybe define a different behaviour for vectors and matrices?
	# this would then return a vector
	@test sim(GenerativeModels.loglikelihoodopt(fill(0.0,5,5),fill(0.0,5,5)), 0.0)
    @test sim(GenerativeModels.loglikelihoodopt(fill(3,5,5),fill(3,5,5)), 0.0)
    @test sim(GenerativeModels.loglikelihoodopt(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5,5)), 0.0)
	@test sim(GenerativeModels.loglikelihoodopt(fill(5,5,5),fill(5,5,5),fill(1,5,5)), 0.0)

	# the version where sigma is a vector (scalar variance)
    @test sim(GenerativeModels.loglikelihoodopt(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5)), 0.0)
	@test sim(GenerativeModels.loglikelihoodopt(fill(5,5,5),fill(5,5,5),fill(1,5)), 0.0)

	# GAN tests

	# MMD tests

	# encoding
	xdim = 5
	ldim = 1
	N = 10
	x = randn(GenerativeModels.Float,xdim,N)
	model = GenerativeModels.AE([xdim,2,ldim], [ldim,2,xdim])
	@test typeof(GenerativeModels.encode(model,x)) <: TrackedArray
	@test !(typeof(GenerativeModels.encode_untracked(model,x)) <: TrackedArray)
	@test typeof(GenerativeModels.encode(model,x,2)) <: TrackedArray
	@test !(typeof(GenerativeModels.encode_untracked(model,x,3)) <: TrackedArray)


	# mu&sigma
	X = randn(4,10)
	@test size(GenerativeModels.mu(X)) == (2,10)
	@test size(GenerativeModels.sigma2(X)) == (2,10)
	@test all(x->x>0, GenerativeModels.sigma2(X)) 

	# mu&sigma scalar
	@test size(GenerativeModels.mu_scalarvar(X)) == (3,10)
	@test size(GenerativeModels.sigma2_scalarvar(X)) == (10,)
	@test all(x->x>0, GenerativeModels.sigma2_scalarvar(X)) 
	
	# sample normal
	M = fill(2,1000)
	sd = fill(0.1,1000)
	X = GenerativeModels.samplenormal(M,sd)
	@test size(X) == (1000,)
	@test sim(StatsBase.mean(X), 2, 1e-2)
	
	M = fill(2,10,1000)
	sd = fill(0.1,1000)
	X = GenerativeModels.samplenormal(M,sd)
	@test size(X) == (10,1000)
	@test sim(StatsBase.mean(X), 2, 1e-1)
	
	X = randn(4,10)
	y = GenerativeModels.samplenormal(X)
	@test size(y) == (2,10)

	# sample normal scalarvar
	X = randn(4,10)
	y = GenerativeModels.samplenormal_scalarvar(X)
	@test size(y) == (3,10)
end