using GenerativeModels
using Test

M = 2
N = 9
Xs = [rand(N), rand(M,N), rand(M,M,N), rand(M,M,M,N)]
uniter = 3
ubatchsize = 4
batchsize = 5
nepochs = 10

@testset "samplers" begin
	@test GenerativeModels.checkbatchsize(10, 3, true) == 3
	@test GenerativeModels.checkbatchsize(10, 3, false) == 3
	@test GenerativeModels.checkbatchsize(3, 10, true) == 10
	@test GenerativeModels.checkbatchsize(3, 10, false) == 3
	
	for X in Xs
		usampler = GenerativeModels.UniformSampler(X, uniter, ubatchsize)
		@test usampler.data == X 
		@test usampler.M == ndims(X)
		@test usampler.N == N
		@test usampler.niter == uniter
		@test usampler.batchsize == ubatchsize
		@test usampler.iter == 0
		@test usampler.replace == false
		batch = GenerativeModels.next!(usampler)
		@test usampler.iter == 1
		Xsize = collect(size(X))
		Xsize[end] = ubatchsize 
		@test collect(size(batch)) == Xsize
		ixs = GenerativeModels.enumerate(usampler)
		@test usampler.iter == uniter
		@test length(ixs) == uniter - 1
		@test collect(size(ixs[1][2])) == Xsize
		@test collect(size(ixs[2][2])) == Xsize
		GenerativeModels.reset!(usampler)
		xs = GenerativeModels.collect(usampler)
		@test length(xs) == uniter
		@test collect(size(xs[1])) == Xsize
		@test collect(size(xs[end])) == Xsize
	end

	for X in Xs
		esampler = GenerativeModels.EpochSampler(X, nepochs, batchsize)
		@test esampler.M == ndims(X)
		@test esampler.N == N
		@test esampler.batchsize == batchsize
		@test esampler.nepochs == nepochs
		@test esampler.data == X
		@test esampler.iter == 0
		@test esampler.epochsize == ceil(Int,N/batchsize)
		batch = GenerativeModels.next!(esampler)
		@test esampler.iter == 0
		Xsize = collect(size(X))
		Xsize[end] = batchsize
		Xsizetrunc = copy(Xsize)
		Xsizetrunc[end] = N%batchsize
		@test collect(size(batch)) == Xsize
		batch = GenerativeModels.next!(esampler)
		@test collect(size(batch)) == Xsizetrunc
		batch = GenerativeModels.next!(esampler)
		@test collect(size(batch)) == Xsize
		ixs = GenerativeModels.enumerate(esampler)
		@test esampler.iter == nepochs
		@test length(ixs) == (nepochs-1)*esampler.epochsize-1
		@test collect(size(ixs[1][2])) == Xsizetrunc
		@test collect(size(ixs[2][2])) == Xsize
		GenerativeModels.reset!(esampler)
		xs = GenerativeModels.collect(esampler)
		@test length(xs) == nepochs*esampler.epochsize
		@test collect(size(xs[1])) == Xsize
		@test collect(size(xs[2])) == Xsizetrunc
	end
end