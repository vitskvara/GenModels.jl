using Flux
using CuArrays
#using BenchmarkTools

X = randn(Float32, 64,64,1,64) |> gpu

model = Flux.Chain(
	Flux.Conv((3,3), 1=>16, pad=(1,1)),
	Flux.Conv((3,3), 16=>1, pad=(1,1))
	) |> gpu

function update!(model, optimiser)
    for p in params(model)
        Δ = Flux.Optimise.apply!(optimiser, p.data, p.grad)
        p.data .-= Δ
        p.grad .= 0
    end
end
	
opt = ADAM()
@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)
sleep(0.5)
@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)


# old version
julia> pathof(Flux.NNlib)
"/home/vit/.julia/packages/NNlib/mxWRT/src/NNlib.jl"

# CPU version
julia> @time l = Flux.mse(model(X),X)
  3.312647 seconds (10.94 M allocations: 626.418 MiB, 8.14% gc time)
11.49951f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  3.925870 seconds (12.63 M allocations: 703.349 MiB, 9.85% gc time)

julia> @time update!(model, opt)
  1.754735 seconds (5.90 M allocations: 302.296 MiB, 5.48% gc time)

julia> @time l = Flux.mse(model(X),X)
  0.103824 seconds (1.43 k allocations: 37.448 MiB, 2.62% gc time)
11.178073f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  0.409683 seconds (2.81 k allocations: 61.895 MiB, 5.67% gc time)

julia> @time update!(model, opt)
  0.000199 seconds (248 allocations: 10.203 KiB)

# GPU version
julia> opt = ADAM()
ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}())

julia> @time l = Flux.mse(model(X),X)
  5.267287 seconds (13.22 M allocations: 706.632 MiB, 6.29% gc time)
6.272906f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  5.828294 seconds (17.44 M allocations: 893.435 MiB, 7.65% gc time)

julia> @time update!(model, opt)
  5.003732 seconds (12.16 M allocations: 641.099 MiB, 4.67% gc time)

julia> @time l = Flux.mse(model(X),X)
  0.025697 seconds (676 allocations: 31.594 KiB)
6.06815f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  0.013474 seconds (1.83 k allocations: 92.688 KiB)

julia> @time update!(model, opt)
  0.000984 seconds (1.51 k allocations: 79.891 KiB)

# using ConvAE
using GenModels

model=GenModels.ConvAE((64,64,1), 2, 2, 3, (16,1),1) |> gpu

opt = ADAM()
@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)

@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)

# GPU run 
julia> @time l = Flux.mse(model(X),X)
  4.053405 seconds (10.54 M allocations: 554.110 MiB, 5.95% gc time)
0.99937373f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  6.242859 seconds (18.21 M allocations: 921.742 MiB, 8.06% gc time)

julia> @time update!(model, opt)
  3.765710 seconds (9.17 M allocations: 470.231 MiB, 5.02% gc time)

julia> @time l = Flux.mse(model(X),X)
  0.075422 seconds (2.06 k allocations: 93.953 KiB)
0.99911374f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  0.008570 seconds (5.41 k allocations: 262.969 KiB)

julia> @time update!(model, opt)
  0.002965 seconds (5.93 k allocations: 303.016 KiB)

# larger model
model=GenModels.ConvAE((64,64,1), 2, 3, 3, (8,16,32),2) |> gpu

opt = ADAM()
@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)

@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)

# it seems to be much faster today... wtf?
julia> @time l = Flux.mse(model(X),X)
  2.353923 seconds (7.65 M allocations: 408.748 MiB, 7.60% gc time)
15.797068f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  1.919192 seconds (5.76 M allocations: 292.349 MiB, 6.11% gc time)

julia> @time update!(model, opt)
  0.088686 seconds (219.27 k allocations: 10.871 MiB)

julia> @time l = Flux.mse(model(X),X)
  0.042764 seconds (2.84 k allocations: 131.906 KiB)
2.9866786f0 (tracked)

julia> @time Flux.Tracker.back!(l)
  0.021240 seconds (7.38 k allocations: 359.500 KiB)

julia> @time update!(model, opt)
  0.003847 seconds (8.16 k allocations: 421.906 KiB)

# larger data
X = randn(Float32, 128, 128, 1, 64) |> gpu;
model=GenModels.ConvAE((128,128,1), 2, 3, 3, (8,16,32),2) |> gpu;

opt = ADAM()
@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)

@time l = Flux.mse(model(X),X)
@time Flux.Tracker.back!(l)
@time update!(model, opt)
