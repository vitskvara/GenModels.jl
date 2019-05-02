using Flux
using PyPlot
using GenerativeModels
using StatsBase
using Random
using ValueHistories

# artificial data
m1 = [-1.0;-1.0]
m2 = [1.0;1.0]
v1 = v2 = 0.1
N1 = N2 = 50
N = N1+N2
M = 2
X = hcat(randn(M,N1)*v1 .+ m1, randn(M,N2)*v2 .+ m2)
X = hcat(X, randn(M,N1)*v1 .+ [-1,1])

# construct the generating distribution
function quadnormal(T::DataType,m::Int)
	@assert m>=2
	i = rand(1:4)
	μ = T.(hcat(fill(-1.0,m), fill(-1.0,m), fill(1.0,m), fill(1.0,m)))
	μ[1,1] = 1.0
	μ[1,3] = -1.0
	σ = T(0.1)
	return randn(T,m)*σ .+ μ[:,i]
end
quadnormal(m::Int) = quadnormal(Float32,m)
quadnormal(T::DataType,m::Int,n::Int) = hcat(map(x->quadnormal(T,m),1:n)...)
quadnormal(m::Int,n::Int) = quadnormal(Float32,m,n)

# construct the generating distribution
function binormal(T::DataType,m::Int)
	i = rand(1:2)
	μ = T.([-1.0f0, 1.0f0])
	σ = T(0.1)
	return randn(T,m)*σ .+ μ[i]
end
binormal(m::Int) = binormal(Float32,m)
binormal(T::DataType,m::Int,n::Int) = hcat(map(x->binormal(T,m),1:n)...)
binormal(m::Int,n::Int) = binormal(Float32,m,n)

# define the parts of the network
modelname = "AAE"
ldim = 2
hdim = 50
nlayers = 3
nonlinearity = Flux.relu
pz = binormal
if model == "WAE"
	kernel = GenerativeModels.imq
	model = WAE(M, ldim, nlayers, pz, kernel = kernel, hdim = hdim, activation=nonlinearity)
elseif modelname == "AAE"
	model = AAE(M, ldim, nlayers, nlayers, pz, hdim = hdim, activation=nonlinearity)
end
hist = MVHistory()
if modelname == "WAE"
	σ = 0.001
	λ = 1.0
	GenerativeModels.fit!(model, X, 50, 2000, σ=σ, λ=λ, history=hist,verb=true);
else
	GenerativeModels.fit!(model, X, 50, 2000, history=hist,verb=true);
end

rX = model(X).data
figure(figsize=(10,5))
subplot(1,2,1)
title("original data and reconstructions")
scatter(X[1,:],X[2,:])
scatter(rX[1,1:N1],rX[2,1:N1],c="r")
scatter(rX[1,N1+1:end],rX[2,N1+1:end],c="g")

Z = model.encoder(X).data
subplot(1,2,2)
title("distribution of latent code")
if ldim == 1
	plt.hist(vec(Z[1:N1]),20,color="r")
	plt.hist(vec(Z[N1+1:end]),20,color="g")
else
	plt.hist2d(Z[1,:],Z[2,:],32)
end
show()

