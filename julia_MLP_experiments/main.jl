using Statistics, Logging, CUDA, Dates, MLDatasets, ProgressMeter, LinearAlgebra, Images
using Random
using LinearAlgebra: tr, diagm
using BSON: @save, @load
using Flux
using Flux: onehotbatch, onecold
using Flux.Data: DataLoader
using Flux.Losses: mse
using Plots; using Plots.PlotMeasures; ENV["GKSwstype"] = "100" # Prevents some warnings when plotting via ssh
using Base: @kwdef

include("utils.jl")
include("dualprop_model.jl")

hiddens = [1000 for i=1:4]
N = [784, hiddens..., 10] # Network architecture: neurons per layer
seeds = [139, 946, 992, 640, 731]
num_epochs = 100
lr = 3e-5
batchsize = 100

# valid options are "backprop", "dualprop", "corrupted_dualprop", "multistep_dualprop", "dualprop_betaL_lin_mse", "parallel_dualprop" and "random_dualprop"
learnmode = "dualprop" 

# Train an MLP with dual propagation
for seed in seeds
    Random.seed!(seed)
    experiment_dir = "test_$(learnmode)/"
    _, metrics = train(experiment_dir; learnmode=learnmode, model=build_dualprop_MLP(N), epochs=num_epochs, opt=Adam(lr), numiter=1, use_cuda=true, γ=1f0, batchsize=batchsize, saveall=false, multistep=0);
    println(metrics["test_acc"][1])
end