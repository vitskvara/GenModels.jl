import Flux.params

"""
	WAAE{encoder, decoder, discriminator, pz, kernel}

Flux-like structure of the wasserstein adversarial autoencoder.
"""
mutable struct WAAE{E, FE, DE, DS, FDS, PZ, K} <: GenerativeModel
	encoder::E 
	f_encoder::FE # frozen encoder copy
	decoder::DE
	discriminator::DS
	f_discriminator::FDS # frozen discriminator copy
	pz::PZ
	kernel::K
end

"""
	WAAE(e, de, ds, pz, k)

Construct AAE given encoder, decoder, discriminator, kernel and pz(n), where n is number of samples and
is compatible with the dimension of encoder output.
"""
WAAE(e, de, ds, pz, k) = WAAE(e, freeze(e), de, ds, freeze(ds), pz, k) # default constructor 

# make the struct callable
(waae::WAAE)(X) = waae.decoder(waae.encoder(X))

# and make it trainable
Flux.@treelike WAAE

"""
	WAAE(esize, decsize, dissize[, pz]; [kernel, activation, layer])

Initialize a wasserstein adversarial autoencoder.

	esize = vector of ints specifying the width anf number of layers of the encoder
	decsize = size of decoder
	dissize = size of discriminator
	pz = sampling distribution that can be called as pz(nsamples)
	kernel = default rbf kernel
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
"""
function WAAE(esize::Array{Int64,1}, decsize::Array{Int64,1}, dissize::Array{Int64,1}, 
	pz = n -> randn(Float,esize[end],n); kernel=rbf, activation = Flux.relu,	layer = Flux.Dense)
	@assert size(esize, 1) >= 3
	@assert size(decsize, 1) >= 3
	@assert size(dissize, 1) >= 3
	@assert esize[end] == decsize[1] 
	@assert esize[1] == decsize[end]
	@assert esize[end] == dissize[1]
	@assert 1 == dissize[end]	

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(decsize, activation, layer)

	# cinstruct the discriminator
	discriminator = discriminatorbuilder(dissize, activation, layer)

	# finally construct the ae struct
	waae = WAAE(encoder, decoder, discriminator, pz, kernel)

	return waae
end

"""
	WAAE(xdim, zdim, ae_nlayers, disc_nlayers[, pz]; [kernel, hdim, activation, layer])

Initialize a wasserstein adversarial autoencoder given input and latent dimension 
and number of layers.

	xdim = input size
	zdim = code size
	ae_nlayers = number of layers of the autoencoder
	disc_nlayers = number of layers of the discriminator
	pz = sampling distribution that can be called as pz(nsamples)
	kernel = default rbf kernel
	hdim = width of layers, if not specified, it is linearly interpolated
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
"""
function WAAE(xdim::Int, zdim::Int, ae_nlayers::Int, disc_nlayers::Int, 
	pz = n -> randn(Float,zdim,n); kernel=rbf, hdim = nothing, 
	activation = Flux.relu, layer = Flux.Dense)
	@assert ae_nlayers >= 2
	@assert disc_nlayers >= 2

	# this will create the integer array specifying the width of individual layers using linear interpolations
	if hdim == nothing
		esize = ceil.(Int, range(xdim, zdim, length=ae_nlayers+1))
		dissize = ceil.(Int, range(zdim, 1, length=disc_nlayers+1))
	else
		esize = vcat([xdim], fill(hdim, ae_nlayers-1), [zdim])
		dissize = vcat([zdim], fill(hdim, disc_nlayers-1), [1])
	end
	decsize = reverse(esize)
	
	# finally return the structure
	WAAE(esize,decsize, dissize, pz; kernel=kernel, activation=activation, layer=layer)
end

"""
	ConvWAAE(insize, zdim, disc_nlayers, nconv, kernelsize, channels, scaling[, pz]; 
		[kernel, hdim, ndense, dsizes, activation, stride, batchnorm, outbatchnorm, upscale_type])

Initialize a convolutional adversarial autoencoder. 
	
	insize = tuple of (height, width, channels)
	zdim = size of latent space
	disc_nlayers = number of layers od the discriminator
	nconv = number of convolutional layers
	kernelsize = Int or a tuple/vector of ints
	channels = a tuple/vector of number of channels
	scaling = Int or a tuple/vector of ints
	pz = sampling distribution that can be called as pz(nsamples)
	kernel = default rbf kernel
	hdim = widht of layers in the discriminator
	ndense = number of dense layers
	dsizes = vector of dense layer widths
	activation = type of nonlinearity
	stride = Int or vecotr/tuple of ints
	batchnorm = use batchnorm in convolutional layers
	outbatchnorm = use batchnorm on the outpu of encoder
	upscale_type = one of ["transpose", "upscale"]
"""
function ConvWAAE(insize, zdim, disc_nlayers, nconv, kernelsize, channels, scaling, 
	pz = n -> randn(Float,zdim,n); 
	outbatchnorm=false, hdim=nothing, activation=Flux.relu, layer=Flux.Dense, upscale_type = "transpose",
	kernel=rbf, kwargs...)
	# first build the convolutional encoder and decoder
	encoder = convencoder(insize, zdim, nconv, kernelsize, 
		channels, scaling; outbatchnorm=outbatchnorm, activation = activation, kwargs...)
	decoder = convdecoder(insize, zdim, nconv, kernelsize, 
		reverse(channels), scaling; layertype = upscale_type, activation = activation, kwargs...)
	# then a classical discriminator
	if hdim == nothing
		dissize = ceil.(Int, range(zdim, 1, length=disc_nlayers+1))
	else
		dissize = vcat([zdim], fill(hdim, disc_nlayers-1), [1])
	end
	discriminator = discriminatorbuilder(dissize, activation, layer)
	return WAAE(encoder, decoder, discriminator, pz, kernel)
end

################
### training ###
################

"""
	aeloss(WAAE, X)

Autoencoder loss.
"""
aeloss(waae::WAAE,X) = Flux.mse(X,waae(X))

"""
	dloss(WAAE,X[,Z])

Discriminator loss given code Z and original sample X. If Z not given, 
it is autoamtically generated using the prescribed pz.
"""
dloss(waae::WAAE,X,Z) = dloss(waae.discriminator, waae.f_encoder, Z, X)
dloss(waae::WAAE,X) = dloss(waae, X, waae.pz(size(X,ndims(X))))  
# note that X and Z is swapped here from the normal notation

"""
	gloss(WAAE,X)

Encoder/generator loss.
"""
gloss(waae::WAAE,X) = gloss(waae.f_discriminator, waae.encoder,X)

"""
	MMD(WAAE, X[, Z], σ)

MMD for a given sample X, scaling constant σ. If Z is not given, it is automatically generated 
from the model's pz.
"""
MMD(waae::WAAE, X::AbstractArray, Z::AbstractArray, σ) = MMD(waae.kernel, waae.encoder(X), Z, Float(σ))
MMD(waae::WAAE, X::AbstractArray, σ) = MMD(waae.kernel, waae.encoder(X), waae.pz(size(X,ndims(X))), Float(σ))

"""
	loss(WAAE,X)

WAAE loss.
"""
loss(waae::WAAE, X::AbstractArray, Z::AbstractArray, σ, λ::Real, γ::Real) = aeloss(waae, X) + Float(λ)*MMD(waae, X, Z, σ) + Float(γ)*gloss(waae,X) + Float(γ)*dloss(waae,X)
loss(waae::WAAE, X::AbstractArray, σ, λ::Real, γ::Real) = aeloss(waae, X) + Float(λ)*MMD(waae, X, σ) + Float(γ)*gloss(waae,X) + Float(γ)*dloss(waae,X)

"""
	getlosses(WAAE, X[, Z], σ, λ)

Return the numeric values of current losses.
"""
getlosses(waae::WAAE, X::AbstractArray, Z::AbstractArray, σ, λ::Real, γ::Real) =  (
		Flux.Tracker.data(loss(waae,X,Z,σ,λ,γ)),
		Flux.Tracker.data(aeloss(waae,X)),
		Flux.Tracker.data(dloss(waae,X,Z)),
		Flux.Tracker.data(gloss(waae,X)),
		Flux.Tracker.data(MMD(waae,X,Z,σ))
		)
getlosses(waae::WAAE, X::AbstractArray, σ, λ::Real, γ::Real) = getlosses(waae::WAAE, X, waae.pz(size(X,ndims(X))), σ, λ, γ)

"""
	evalloss(WAAE, X[, Z], σ, λ)

Print WAAE losses.
"""
function evalloss(waae::WAAE, X::AbstractArray, Z::AbstractArray, σ, λ::Real, γ::Real) 
	l, ael, dl, gl, mmd = getlosses(waae, X, Z, σ, λ, γ)
	print("total loss: ", l,
	"\nautoencoder loss: ", ael,
	"\ndiscriminator loss: ", dl,
	"\ngenerator loss: ", gl,
	"\nMMD loss: ", mmd, "\n\n")
end
evalloss(waae::WAAE, X::AbstractArray, σ, λ::Real, γ::Real) = evalloss(waae::WAAE, X, waae.pz(size(X,2)), σ, λ, γ)

"""
	getlsize(WAAE)

Return size of the latent code.
"""
getlsize(waae::WAAE) = size(waae.encoder.layers[end].W,1)

"""
	track!(WAAE, history, X)

Save current progress.
"""
function track!(waae::WAAE, history::MVHistory, X::AbstractArray, σ, λ::Real,γ::Real)
	l, ael, dl, gl, mmd = getlosses(waae, X, σ, λ,γ)
	push!(history, :loss, l)
	push!(history, :aeloss, ael)
	push!(history, :dloss, dl)
	push!(history, :gloss, gl)
	push!(history, :mmd, mmd)
end

"""
	(cb::basic_callback)(WAAE, d, l, opt, σ, λ)

Callback for the train! function.
TODO: stopping condition, change learning rate.
"""
function (cb::basic_callback)(m::WAAE, d, l, opt, σ, λ::Real,γ::Real)
	# update iteration count
	cb.iter_counter += 1
	# save training progress to a MVHistory
	if cb.history != nothing
		track!(m, cb.history, d, σ, λ, γ)
	end
	# verbal output
	if cb.verb 
		# if first iteration or a progress print iteration
		# recalculate the shown values
		if (cb.iter_counter%cb.show_it == 0 || cb.iter_counter == 1)
			ls = getlosses(m, d, σ, λ, γ)
			cb.progress_vals = Array{Any,1}()
			push!(cb.progress_vals, ceil(Int, cb.iter_counter/cb.epoch_size))
			push!(cb.progress_vals, cb.iter_counter)
			push!(cb.progress_vals, ls[1])
			push!(cb.progress_vals, ls[2])
			push!(cb.progress_vals, ls[3])
			push!(cb.progress_vals, ls[4])
			push!(cb.progress_vals, ls[5])
		end
		# now give them to the progress bar object
		ProgressMeter.next!(cb.progress; showvalues = [
			(:epoch,cb.progress_vals[1]),
			(:iteration,cb.progress_vals[2]),
			(:loss,cb.progress_vals[3]),
			(:aeloss,cb.progress_vals[4]),
			(:dloss,cb.progress_vals[5]),
			(:gloss,cb.progress_vals[6]),
			(:mmd,cb.progress_vals[7])
			])
	end
end

"""
	fit!(WAAE, X, batchsize, nepochs; 
		[σ, λ, γ, cbit, history, verb, η, runtype, usegpu, memoryefficient])

Trains the WAAE neural net.

	WAAE - a WAAE object
	X - data array with instances as columns
	batchsize - batchsize
	nepochs - number of epochs
	σ - scaling parameter of the MMD
	λ - scaling for the MMD loss
	γ - scaling for the GAN loss
	cbit [200] - after this # of iterations, progress is updated
	history [nothing] - a dictionary for training progress control
	verb [true] - if output should be produced
	η [0.001] - learning rate
	runtype ["experimental"] - if fast is selected, no output and no history is written
	usegpu - if X is not already on gpu, this will put the inidvidual batches into gpu memory rather 
			than all data at once
	memoryefficient - calls gc after every batch, again saving some memory but prolonging computation
"""
function fit!(waae::WAAE, X, batchsize::Int, nepochs::Int; 
	σ::Real=1.0, λ::Real=1.0, γ::Real=1.0, cbit::Int=200, history = nothing, opt=nothing,
	verb::Bool = true, η = 0.001, runtype = "experimental", trainkwargs...)
	@assert runtype in ["experimental", "fast"]
	# sampler
	sampler = EpochSampler(X,nepochs,batchsize)
	epochsize = sampler.epochsize
	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# losses
	
	# optimizer
	if opt == nothing
		opt = ADAM(η)
	end
	
	# callback
	if runtype == "experimental"
		cb = basic_callback(history,verb,η,cbit; 
			train_length = nepochs*epochsize,
			epoch_size = epochsize)
		_cb(m::WAAE,d,l,o) =  cb(m,d,l,o,σ,λ,γ)
	elseif runtype == "fast"
		_cb = fast_callback 
	end

	# preallocate arrays?

	# train
	train!(
		waae,
		collect(sampler),
		x->loss(waae,x,σ,λ,γ),
		opt,
		_cb;
		trainkwargs...
		)

	return opt
end


"""
	sample(WAAE[, M])

Get samples generated by the WAAE.
"""
StatsBase.sample(waae::WAAE) = waae.decoder(waae.pz(1))
StatsBase.sample(waae::WAAE, M) = waae.decoder(waae.pz(M))