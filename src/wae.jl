"""
	WAE{encoder, decoder, pz, kernel}

Flux-like structure for the Wasserstein autoencoder with MMD loss.
"""
struct WAE{E, D, P, K}
	encoder::E
	decoder::D
	pz::P
	kernel::K
end

# and make it trainable
Flux.@treelike WAE

# make the struct callable
(wae::WAE)(X) = wae.decoder(wae.encoder(X))

"""
	WAE(esize, dsize[, pz]; [kernel, activation, layer])

Initialize a Wasserstein autoencoder with given encoder size and decoder size.

	esize = vector of ints specifying the width anf number of layers of the encoder
	dsize = size of decoder
	pz = generating distribution that can be called as pz(T,dim,nsamples)
	kernel = default rbf kernel
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = type of layer
"""
function WAE(esize::Array{Int64,1}, dsize::Array{Int64,1}, pz=randn; kernel = rbf,
		activation = Flux.relu, layer = Flux.Dense)
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == dsize[1]
	@assert esize[1] == dsize[end]

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(dsize, activation, layer)

	# finally construct the ae struct
	wae = WAE(encoder, decoder, n->pz(Float,esize[end],n), kernel)

	return wae
end

"""
	WAE(xdim, zdim, nlayers[, pz]; [kernel, hdim, activation, layer])

Initialize a variational autoencoder given input and latent dimension 
and numberof layers. The width of layers is linearly interpolated 
between xdim and zdim.

	xdim = input size
	zdim = code size
	nlayers = number of layers
	pz = generating distribution that can be called as pz(T,dim,nsamples)
	kernel = default rbf kernel
	hdim = width of layers, if not specified, it is linearly interpolated
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
"""
function WAE(xdim::Int, zdim::Int, nlayers::Int, pz=randn; kernel=rbf, 
		activation = Flux.relu,	hdim = nothing, layer = Flux.Dense)
	@assert nlayers >= 2

	if hdim == nothing
		esize = ceil.(Int, range(xdim, zdim, length=nlayers+1))
	else
		esize = vcat([xdim], fill(hdim, nlayers-1), [zdim])
	end
	dsize = reverse(esize)
	
	WAE(esize,dsize, pz; kernel=kernel, activation=activation, layer=layer)
end

"""
	ConvWAE(insize, zdim, nconv, kernelsize, channels, scaling[, pz]; 
		[kernel, ndense, dsizes, activation, stride, batchnorm, outbatchnorm])

Initialize a convolutional wasserstein autoencoder. 
	
	insize = tuple of (height, width, channels)
	zdim = size of latent space
	nconv = number of convolutional layers
	kernelsize = Int or a tuple/vector of ints
	channels = a tuple/vector of number of channels
	scaling = Int or a tuple/vector of ints
	pz = sampling distribution that can be called as pz(T,dim,nsamples)
	kernel = default rbf kernel
	ndense = number of dense layers
	dsizes = vector of dense layer widths
	activation = type of nonlinearity
	stride = Int or vecotr/tuple of ints
	batchnorm = use batchnorm in convolutional layers
	outbatchnorm = use batchnorm on the outpu of encoder
"""
function ConvWAE(insize, zdim, nconv, kernelsize, channels, scaling, pz=randn; 
	kernel = rbf, outbatchnorm=false, kwargs...)
	# first build the convolutional encoder and decoder
	encoder = convencoder(insize, zdim, nconv, kernelsize, 
		channels, scaling; outbatchnorm=outbatchnorm, kwargs...)
	decoder = convdecoder(insize, zdim, nconv, kernelsize, 
		reverse(channels), scaling; kwargs...)

	return WAE(encoder, decoder, n->pz(Float,zdim,n), kernel)
end

###########################
### losses and training ###
###########################

"""
	aeloss(WAE, X)

Autoencoder reconstruction loss.
"""
aeloss(wae::WAE, X) = Flux.mse(wae(X), X)

"""
	MMD(WAE, X[, Z], c)

MMD for a given sample X, scaling constant c. If Z is not given, it is automatically generated 
from the model's pz.
"""
MMD(wae::WAE, X::AbstractArray, Z::AbstractArray, c) = MMD(wae.kernel, wae.encoder(X), Z, Float(c))
MMD(wae::WAE, X::AbstractArray, c) = MMD(wae.kernel, wae.encoder(X), wae.pz(size(X,2)), Float(c))

"""
	loss(WAE, X[, Z], c, λ)

Total loss of the WAE. λ is the scaling parameter of the MMD in the total loss.
"""
loss(wae::WAE, X::AbstractArray, Z::AbstractArray, c, λ::Real) = aeloss(wae, X) + Float(λ)*MMD(wae, X, Z, c)
loss(wae::WAE, X::AbstractArray, c, λ::Real) = aeloss(wae, X) + Float(λ)*MMD(wae, X, c)

"""
	getlosses(WAE, X[, Z], c, λ)

Obtain the nuemrical values of the WAE losses.
"""
getlosses(wae::WAE, X::AbstractArray, Z::AbstractArray, c, λ::Real) =  (
		Flux.Tracker.data(loss(wae,X,Z,c,λ)),
		Flux.Tracker.data(aeloss(wae,X)),
		Flux.Tracker.data(MMD(wae,X,Z,c))
		)
getlosses(wae::WAE, X::AbstractArray, c, λ::Real) = getlosses(wae::WAE, X, wae.pz(size(X,2)), c, λ)

"""
	evalloss(WAE, X[, Z], c, λ)

Print WAE losses.
"""
function evalloss(wae::WAE, X::AbstractArray, Z::AbstractArray, c, λ::Real) 
	l, ael, mmd = getlosses(wae, X, Z, c, λ)
	print("total loss: ", l,
	"\nautoencoder loss: ", ael,
	"\nMMD loss: ", mmd, "\n\n")
end
evalloss(wae::WAE, X::AbstractArray, c, λ::Real) = evalloss(wae::WAE, X, wae.pz(size(X,2)), c, λ)

"""
	getlsize(WAE)

Return size of the latent code.
"""
getlsize(wae::WAE) = size(wae.encoder.layers[end].W,1)

"""
	track!(WAE, history, X, c)

Save current progress.
"""
function track!(wae::WAE, history::MVHistory, X::AbstractArray, c, λ::Real)
	l, ael, mmd = getlosses(wae, X, c, λ)
	push!(history, :loss, l)
	push!(history, :aeloss, ael)
	push!(history, :mmd, mmd)
end

"""
	(cb::basic_callback)(WAE, d, l, opt, c, λ)

Callback for the train! function.
TODO: stopping condition, change learning rate.
"""
function (cb::basic_callback)(m::WAE, d, l, opt, c, λ::Real)
	# update iteration count
	cb.iter_counter += 1
	# save training progress to a MVHistory
	if cb.history != nothing
		track!(m, cb.history, d, c, λ)
	end
	# verbal output
	if cb.verb 
		# if first iteration or a progress print iteration
		# recalculate the shown values
		if (cb.iter_counter%cb.show_it == 0 || cb.iter_counter == 1)
			ls = getlosses(m, d, c, λ)
			cb.progress_vals = Array{Any,1}()
			push!(cb.progress_vals, ceil(Int, cb.iter_counter/cb.epoch_size))
			push!(cb.progress_vals, cb.iter_counter)
			push!(cb.progress_vals, ls[1])
			push!(cb.progress_vals, ls[2])
			push!(cb.progress_vals, ls[3])
		end
		# now give them to the progress bar object
		ProgressMeter.next!(cb.progress; showvalues = [
			(:epoch,cb.progress_vals[1]),
			(:iteration,cb.progress_vals[2]),
			(:loss,cb.progress_vals[3]),
			(:aeloss,cb.progress_vals[4]),
			(:mmd,cb.progress_vals[5])
			])
	end
end

"""
	fit!(WAE, X, batchsize, nepochs; 
		[c, λ, cbit, history, verb, η, runtype, usegpu, memoryefficient])

Trains the WAE neural net.

	WAE - a WAE object
	X - data array with instances as columns
	batchsize - batchsize
	nepochs - number of epochs
	c - scaling parameter of the MMD
	λ - scaling for the MMD loss
	cbit [200] - after this # of iterations, progress is updated
	history [nothing] - a dictionary for training progress control
	verb [true] - if output should be produced
	η [0.001] - learning rate
	runtype ["experimental"] - if fast is selected, no output and no history is written
	usegpu - if X is not already on gpu, this will put the inidvidual batches into gpu memory rather 
			than all data at once
	memoryefficient - calls gc after every batch, again saving some memory but prolonging computation
"""
function fit!(wae::WAE, X, batchsize::Int, nepochs::Int; 
	c::Real=1.0, λ::Real=1.0, cbit::Int=200, history = nothing, opt=nothing,
	verb::Bool = true, η = 0.001, runtype = "experimental", trainkwargs...)
	@assert runtype in ["experimental", "fast"]
	# sampler
	sampler = EpochSampler(X,nepochs,batchsize)
	epochsize = sampler.epochsize
	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# loss
	# use default loss

	# optimizer
	if opt == nothing
		opt = ADAM(η)
	end
	
	# callback
	if runtype == "experimental"
		cb = basic_callback(history,verb,η,cbit; 
			train_length = nepochs*epochsize,
			epoch_size = epochsize)
		_cb(m::WAE,d,l,o) =  cb(m,d,l,o,c,λ)
	elseif runtype == "fast"
		_cb = fast_callback 
	end

	# preallocate arrays?

	# train
	train!(
		wae,
		collect(sampler),
		x->loss(wae,x,c,λ),
		opt,
		_cb;
		trainkwargs...
		)

	return opt
end

