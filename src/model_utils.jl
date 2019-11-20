import Base.length

const l2pi = Float(log(2*pi)) # the model converges the same with zero or correct value
const δ = Float(1e-6)
const half = Float(0.5)

"""
    KL(μ, σ2)

KL divergence between a normal distribution and unit gaussian.
"""
KL(μ::Real, σ2::Real) = Float(1/2)*(σ2 + μ^2 - log(σ2) .- Float(1.0))
KL(μ, σ2) = Float(1/2)*StatsBase.mean(sum(σ2 + μ.^2 - log.(σ2) .- Float(1.0), dims = 1))
# have this for the general full covariance matrix normal distribution?

"""
    loglikelihood(X, μ, [σ2])

Loglikelihood of a normal sample X given mean and variance.
"""
loglikelihood(X::Real, μ::Real) = - ((μ - X)^2 + l2pi)*half
loglikelihood(X::Real, μ::Real, σ2::Real) = - ((μ - X)^2/σ2 + log(σ2) + l2pi)*half
loglikelihood(X, μ) = - StatsBase.mean(sum((μ - X).^2 .+ l2pi,dims = 1))*half
loglikelihood(X, μ, σ2) = - StatsBase.mean(sum((μ - X).^2 ./σ2 + log.(σ2) .+ l2pi,dims = 1))*half
# in order to work on gpu and for faster backpropagation, dont use .+ here for arrays
# see also https://github.com/FluxML/Flux.jl/issues/385
function loglikelihood(X::AbstractMatrix, μ::AbstractMatrix, σ2::AbstractVector) 
    # again, this has to be split otherwise it is very slow
    y = (μ - X).^2
    y = (one(Float) ./σ2)' .* y 
    - StatsBase.mean(sum( y .+ reshape(log.(σ2), 1, length(σ2)) .+ l2pi,dims = 1))*half
end

"""
    loglikelihoodopt(X, μ, [σ2])

Loglikelihood of a normal sample X given mean and variance without the constant term. For
optimalization the results is the same and this is faster.
"""
loglikelihoodopt(X::Real, μ::Real) = - ((μ - X)^2)*half
loglikelihoodopt(X::Real, μ::Real, σ2::Real) = - ((μ - X)^2/σ2 + log(σ2))*half
loglikelihoodopt(X, μ) = - StatsBase.mean(sum((μ - X).^2,dims = 1))*half
loglikelihoodopt(X, μ, σ2) = - StatsBase.mean(sum( (μ - X).^2 ./σ2 + log.(σ2),dims = 1))*half
function loglikelihoodopt(X::AbstractArray{T,4}, μ::AbstractArray{T,4}, σ2::AbstractArray{T,4}) where T
    y = (μ - X).^2
    y = y ./ σ2
    - StatsBase.mean(sum( y .+ log.(σ2),dims = 1))*half
end
# in order to work on gpu and for faster backpropagation, dont use .+ here
# see also https://github.com/FluxML/Flux.jl/issues/385
function loglikelihoodopt(X::AbstractMatrix, μ::AbstractMatrix, σ2::AbstractVector) 
    # again, this has to be split otherwise it is very slow
    y = (μ - X).^2
    y = (one(Float) ./σ2)' .* y 
    - StatsBase.mean(sum( y .+ reshape(log.(σ2), 1, length(σ2)),dims = 1))*half
end

"""
    mu(X)

Extract mean as the first horizontal half of X.
"""
mu(X) = X[1:Int(size(X,1)/2),:]
mu(X::AbstractArray{T,4}) where T = X[:,:,1:Int(size(X,3)/2),:]

"""
    mu_scalarvar(X)

Extract mean as all but the last rows of X.
"""
mu_scalarvar(X) = X[1:end-1,:]
mu_scalarvar(X::AbstractArray{T,4}) where T = X[:,:,1:Int(size(X,3)/2),:]

"""
    sigma2(X)

Extract sigma^2 as the second horizontal half of X. 
"""
sigma2(X) = softplus.(X[Int(size(X,1)/2+1):end,:]) .+ δ
sigma2(X::AbstractArray{T,4}) where T = softplus.(X[:,:,Int(size(X,3)/2+1):end,:]) .+ δ

"""
    sigma2_scalarvar(X)

Extract sigma^2 as the last row of X. 
"""
sigma2_scalarvar(X) = softplus.(X[end,:]) .+ δ
sigma2_scalarvar(X::AbstractArray{T,4}) where T = StatsBase.mean(softplus.(X[:,:,Int(size(X,3)/2+1):end,:]) .+ δ, dims=[1,2])

"""
    randn_gpu(T,m,n)

GPU version of randn.
"""
randn_gpu(T,m,n) = gpu(randn(T,m,n))

"""
   samplenormal(μ, σ2)

Sample a normal distribution with given mean and standard deviation.
"""
function samplenormal(μ, σ2)
    ϵ = randn(Float, size(μ))    
    # if cuarrays are loaded and X is on GPU, convert eps to GPU as well
    if iscuarray(μ)
        ϵ = ϵ |> gpu
    end
    return μ +  ϵ .* sqrt.(σ2)
end
function samplenormal(μ::AbstractMatrix, σ2::AbstractVector)
    ϵ = randn(Float, size(μ))
    # if cuarrays are loaded and X is on GPU, convert eps to GPU as well
    if iscuarray(μ)
        ϵ = ϵ |> gpu
    end
    return μ +  sqrt.(σ2)' .* ϵ  
end
# version for preallocated ϵ
function samplenormal!(μ, σ2, ϵ)
    randn!(ϵ)
    return μ +  ϵ .* sqrt.(σ2)
end

"""
    samplenormal(X)

Sample normal distribution with mean and sigma2 extracted from X.
"""
function samplenormal(X)
    μ, σ2 = mu(X), sigma2(X)
    return samplenormal(μ, σ2)
end
function samplenormal!(X, ϵ)
    μ, σ2 = mu(X), sigma2(X)
    return samplenormal!(μ, σ2,ϵ)
end

"""
   samplenormal_scalarvar(X)

Sample normal distribution from X where variance is the last row. 
"""
function samplenormal_scalarvar(X)
    μ, σ2 = mu_scalarvar(X), sigma2_scalarvar(X)
    return samplenormal(μ, σ2)
end

length(x::MVHistory) = 1

# GAN losses
"""
    dloss(d,g,X,Z)

Discriminator loss.
"""
dloss(d,g,X,Z) = - half*(mean(log.(d(X) .+ eps(Float))) + mean(log.(1 .- d(g(Z)) .+ eps(Float))))

"""
    gloss(d,g,X)

Generator loss.
"""
gloss(d,g,X) = - mean(log.(d(g(X)) .+ eps(Float)))

# Stuff for WAE and MMD computation
"""
    rbf(x,y,σ)

Gaussian kernel of x and y.
"""
rbf(x,y,σ) = exp.(-(sum((x-y).^2,dims=1)/(2*σ)))

"""
    imq(x,y,σ)

Inverse multiquadratics kernel of x and y.    
"""
imq(x,y,σ) = σ./(σ.+sum(((x-y).^2),dims=1))

"""
    ekxy(k,X,Y,σ)

E_{x in X,y in Y}[k(x,y,σ)] - mean value of kernel k.
"""
ekxy(k,X,Y,σ) = mean(k(X,Y,σ))

"""
    MMD(k,X,Y,σ)

Maximum mean discrepancy for samples X and Y given kernel k and parameter σ.    
"""
MMD(k,X,Y,σ) = ekxy(k,X,X,σ) - 2*ekxy(k,X,Y,σ) + ekxy(k,Y,Y,σ)

"""
    encode(GenerativeModel, X[, batchsize])

Returns the latent space representation of X. If batchsize is specified,
it X will be processed in batches (sometimes necessary due to memory constraints).
"""
encode(model::GenerativeModel, X) = model.encoder(X)
encode_untracked(model::GenerativeModel, X) = Flux.Tracker.data(model.encoder(X))
# general function for both tracked and untracked encoding via limited batchsize
# see that the tracked encoding allocates huge amount of memory for large inputs and conv networks
# made general so that it works both for 2D and 4D inputs
function encode_in_batch(model::GenerativeModel, X, batchsize::Int,enc_fun)
    Z=[]
    Ndim = ndims(X)
    N = size(X,Ndim)
    for i in 1:ceil(Int,N/batchsize)
        inds = Array{Any,1}(fill(Colon(), Ndim-1))
        push!(inds, 1+(i-1)*batchsize:min(i*batchsize, N))
        push!(Z, enc_fun(model, X[inds...]))
    end
    return cat(Z..., dims=ndims(Z[1]))
end
encode(model::GenerativeModel, X, batchsize::Int) = encode_in_batch(model, X, batchsize, encode) 
encode_untracked(model::GenerativeModel, X, batchsize::Int) = encode_in_batch(model, X, batchsize, encode_untracked) 
    
# other auxiliary functions
"""
   scalar2tuple(x)

If x is scalar, return a tuple containing (x,deepcopy(x)). 
"""
function scalar2vec(x)
    if x == nothing
        return Array{Any,1}([nothing,nothing])
    elseif typeof(x) == Symbol
        return Array{Any,1}([x,x])
    elseif length(x) == 1
        return Array{Any,1}([x,deepcopy(x)])
    end
    return x
end