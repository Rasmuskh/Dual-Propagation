# We set default values for learning rate, batch size, epochs, and the usage of a GPU (if available) for the model:
@kwdef mutable struct Args
    model::Chain = build_dualprop_MLP()
    learnmode::String = "dualprop"  # valid options are "backprop", "dualprop", "corrupted_dualprop", "multistep_dualprop", "dualprop_betaL_lin_mse", "parallel_dualprop" and "random_dualprop"
    opt = ADAM(1e-3)        # optimizer
    batchsize::Int = 512    # batch size
    epochs::Int = 20        # number of epochs
    numiter::Int = 2        # number of inference iterations (each iterations loops backwards and forwards through layers)
    use_cuda::Bool = true   # use gpu (if cuda available)
    γ::Float32 = 1.0f0      # feedback gain factor corresponds to βₖ/βₖ₊₁ in the paper
    saveall::Bool = false   # By default only save when new best model is found
    multistep::Int = 0      # multiple inference steps and weight updates per minibatch. only used with learnmode="multistep_dualprop"
end

# pretty printing stuff 🌈
function Base.show(io::IO, a::Args)
    print(io, "Args(model=")
    for l in a.model
        println(io, l, ",")
    end
    print(io, "opt=", a.opt)
    print(io, ", learnmode=", a.learnmode)
    print(io, ", batchsize=", a.batchsize)
    print(io, ", epochs=", a.epochs)
    print(io, ", num_iter=", a.numiter)
    print(io, ", use_cuda=", a.use_cuda)
    print(io, ", γ=", a.γ)
    print(io, ", saveall=", a.saveall)
    print(io, ", multistep=", a.multistep)
    print(io, ")")
end

function angle(a, b)
    return acosd(clamp(dot(a,b)/(norm(a)*norm(b)), -1, 1))
end

function cosine_sim(a, b)
    return clamp(dot(a,b)/(norm(a)*norm(b)), -1, 1)
end

function create_metric_dict(args, keys)
    D = Dict([k => zeros(Float32, args.epochs) for k in keys]...)
    return D
end

round3(x) = round(x, sigdigits=3)

function getdevice(use_cuda)
    if CUDA.functional() && use_cuda
        device = gpu
        @info """Checking hardware
        \tCPU: $(Sys.cpu_info()[1].model)
        \tGPU: $(CUDA.name(CUDA.device()))
        \tTraining on GPU 🚀"""
    else
        device = cpu
        @info """Checking hardware
        \tCPU: $(Sys.cpu_info()[1].model)
        \tGPU: no CUDA capable GPU selected
        \tTraining on CPU 🐢"""
    end
    return device
end

function getdata(args, val_frac=0.1)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    ## Load dataset	
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]
	
    ## Reshape input data to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    maxIndex = Int(round(size(xtrain, 2)*(1-val_frac)))
    xval = xtrain[:, maxIndex+1:end]
    xtrain = xtrain[:, 1:maxIndex]

    xtest = Flux.flatten(xtest)

    ## One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
    yval = ytrain[:, maxIndex+1:end]
    ytrain = ytrain[:, 1:maxIndex]

    ## Create two DataLoader objects (mini-batch iterators)
    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader((xval, yval), batchsize=args.batchsize)

    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, val_loader, test_loader
end

function plot_filters(W, outpath; d=28*4)
    t1 = time()
    filters = [rotr90(reshape(W[i,:], 28, 28), 1) for i=1:64]
    plist = [heatmap(fi, axis = nothing, thickness_scaling=0, size=(d, d), colorbar=false, c = :greys) for fi in filters]
    p = plot(plist..., layout=(8,8), thickness_scaling=0.2, size=(8*d,8*d))
    savefig(p, outpath)
    plottime = time()-t1
    @info "filter snapshot saved: $(round(plottime, sigdigits=3)) s"
end

function plot_metrics(m, outpath, metrics::Dict)
    d=(1600, 200)
    p_acc = plot(metrics["val_acc"], label="test acc")
    plot!(metrics["train_acc"], label="train acc")
    
    p_loss_test = plot(metrics["val_loss"], label="MSE validation loss")
    p_loss_train = plot(metrics["train_loss"], label="MSE train loss")
    p_loss_contr_train = plot(metrics["train_contr_loss"], label="Contrastive train loss")

    p = plot(p_acc, p_loss_test, p_loss_train, p_loss_contr_train, layout=(1, 4), size=d, left_margin = 8mm, bottom_margin = 8mm, lw=3)
    savefig(p, outpath)
end

function plot_grad_cosine_sim(m, outpath, metrics::Dict)
    bm = 16Plots.mm
    lm = 16Plots.mm
    rm = 6Plots.mm

    L = length(m)
    M = [metrics["grad_cosine_sim_$i"]' for i=1:L]
    epochs = length(M[1])
    M = vcat(M...)
    p = heatmap(M, yflip=false, xlabel="Epoch", ylabel="Layer", colorbar_title="Cosine similarity", bottom_margin = bm, left_margin = lm, right_margin = rm, c=:viridis, size=(30*epochs, 45*L))    
    savefig(p, outpath)
end

hs(x) = min(1, max(0, x))

function evaluate(data_loader, model, device, infostring)
    t1 = time()
    acc = 0f0
    ls = 0f0
    num = 0

    for (x, y) in data_loader
        x, y = device(x), device(y)
        num +=  size(x)[end]
        pred = model(x)
        ls += sum(abs2, pred - y)
        acc += sum(onecold(pred) .== onecold(y)) ## Decode the output of the model
    end

    ls /= num
    acc /= num
    runtime = time() - t1
    @info """$infostring: runtime=$(round(runtime, sigdigits=3))s
    \tFF:    loss=$(round(ls, sigdigits=4))\tacc=$(round(acc, sigdigits=4))"""
    return ls, acc
end

function infer_z!(model::Chain, x::AbstractArray, y::AbstractArray, numiter::Int, γ::Float32)
    # t=1: To begin with upstream activity is all zeroes so the first pass reduces to a standard forward pass.
    z⁺ = [x, Flux.activations(model, x)...]
    z⁻ = deepcopy(z⁺); 
    # nudge output units and update neurons in the top down direction
    infer_zL!(model, y, z⁺, z⁻, γ)
    infer_zk_top_down!(model, z⁺, z⁻, γ)

    # t≥2: continue iterating up and down while updating neurons recurrently
    for t=2:numiter
        infer_zk_bottom_up!(model, z⁺, z⁻, γ)
        infer_zL!(model, y, z⁺, z⁻, γ)
        infer_zk_top_down!(model, z⁺, z⁻, γ)
    end

    return z⁺, z⁻
end

function infer_z!(model::Chain, y::AbstractArray, z⁺::AbstractArray, z⁻::AbstractArray, numiter::Int, γ::Float32)
    # Iterate up and down while updating neurons recurrently
    for t=1:numiter
        infer_zk_bottom_up!(model, z⁺, z⁻, γ)
        infer_zL!(model, y, z⁺, z⁻, γ)
        infer_zk_top_down!(model, z⁺, z⁻, γ)
    end

    return z⁺, z⁻
end

function infer_zk_top_down!(model::Chain, z⁺::AbstractArray, z⁻::AbstractArray, γ::Float32)
    for k=length(z⁺)-1:-1:2
        model[k-1](z⁺[k-1], z⁺[k], z⁺[k+1], z⁻[k-1], z⁻[k], z⁻[k+1], γ)
    end
    return z⁺, z⁻
end

function infer_zk_bottom_up!(model::Chain, z⁺::AbstractArray, z⁻::AbstractArray, γ::Float32)
    for k=2:length(z⁺)-1
        model[k-1](z⁺[k-1], z⁺[k], z⁺[k+1], z⁻[k-1], z⁻[k], z⁻[k+1], γ)
    end
    return z⁺, z⁻
end

function infer_zL!(model::Chain, y::AbstractArray, z⁺::AbstractArray, z⁻::AbstractArray, γ::Float32)
    z⁺[end] .= (model[end].Wₖ₋₁*(z⁺[end-1] + z⁻[end-1]) .+ 2*model[end].bₖ₋₁ .+ γ*y)/(2 + γ)
    z⁻[end] .= (model[end].Wₖ₋₁*(z⁺[end-1] + z⁻[end-1]) .+ 2*model[end].bₖ₋₁ .- γ*y)/(2 - γ)
end

function infer_zL_lin_mse!(model::Chain, y::AbstractArray, z⁺::AbstractArray, z⁻::AbstractArray, γ::Float32)
    zstar = 0.5f0*(z⁺[end] + z⁻[end])
    z⁺[end] .= (model[end].Wₖ₋₁*(z⁺[end-1] + z⁻[end-1])) / 2 .+ model[end].bₖ₋₁ .+ γ*(y - zstar)
    z⁻[end] .= (model[end].Wₖ₋₁*(z⁺[end-1] + z⁻[end-1])) / 2 .+ model[end].bₖ₋₁ .- γ*(y - zstar)
end


function train(seed, outpath=nothing; kws...)
    # Collect options in a struct for convenience
    args = Args(; kws...) 
    # Set up Logging
    timestamp = "$(now())"
    if outpath == nothing
        outpath = "output/$(timestamp)_$(args.learnmode)"
    else
        outpath = "output/$(outpath)/$(timestamp)_$(args.learnmode)"
    end
    mkpath(outpath)
    io = open("$outpath/log.txt", "w+")
    logger = ConsoleLogger(io)
    global_logger(logger)
    println("Output directed to experiment log: $(pwd())/$(outpath)")

    # path for saving models
    modelpath = "$outpath/models"
    best_model_path = ""
    println(best_model_path)
    mkpath(modelpath)
    @save "$(outpath)/args.bson" args
    @info "Running seed $seed"
    @info args
    device = getdevice(args.use_cuda)

    # Create dict for storing performance metrics and other useful info
    grad_angle_keys = ["grad_cosine_sim_$i" for i=1:length(args.model)]
    keys = vcat(["train_acc", "val_acc", "train_loss", "val_loss", "train_contr_loss", "runtime"], grad_angle_keys)
    metrics = create_metric_dict(args, keys)

    # Create test and train dataloaders
    train_loader, val_loader, test_loader = getdata(args)

    model = device(args.model)# |> device
    ps = Flux.params(model) # model's trainable parameters

    # network activations
    dummy = device(zeros(Float32, 784, args.batchsize))
    z⁺ = 0f0*[dummy, Flux.activations(model, dummy)...]
    z⁻ = deepcopy(z⁺)


    acc_best = 0f0
    ## Training
    println("🤖Training started🤖:\nTraining model using $(args.learnmode)")
    for epoch in 1:args.epochs
        t1 = time()

        batch_index, num, ls_sum = 0f0, 0f0, 0f0
        grad_cosine_sim = [0f0 for i=1:2:length(ps)]

        @showprogress "Epoch $(epoch)/$(args.epochs)" for (x, y) in train_loader
            x, y = device(x), device(y) ## transfer data to device
            num +=  size(x)[end]
            batch_index += 1
            ls = 0f0

            #compute reference gradient (for computing cosine sim)
            gs_BP = gradient(() -> mse(model(x), y), ps) # For quickly comparing to backprop training
            # If backprop mode then use backprop gradient for learning
            if args.learnmode == "backprop"
                Flux.Optimise.update!(args.opt, ps, gs_BP) # update parameters

            # compute gradient of contrastive objective wrt learnable parameters 
            # z⁺ and z⁻ are treated as constants here, so the computation is fully local (i.e. there is no backpropagation taking place)
            elseif args.learnmode == "dualprop"
                z⁺, z⁻ = infer_z!(model, x, y, args.numiter, args.γ)
                pred = 0.5f0*(z⁺[end] + z⁻[end])

                gs = gradient(ps) do
                    ls = contr_loss_network(model, args.batchsize, z⁺, z⁻, pred, y, args.γ)
                end
                grad_cosine_sim += [cosine_sim(gs[layer.Wₖ₋₁], gs_BP[layer.Wₖ₋₁]) for layer in model]
                Flux.Optimise.update!(args.opt, ps, gs) # update parameters
                ls_sum += ls
            
            
            # Contrary to standard dualprop we do not assume zero activity in upstream neurons, so we can not use a pure FF pass
            # We use the reccurent rule for both forwards and backwards rush, which means activity will be corrupted by feedback from old data.
            elseif args.learnmode == "corrupted_dualprop"
                # notice that we are dispatching to a different version of infer_z! here
                z⁺[1], z⁻[1] = x, x
                z⁺, z⁻ = infer_z!(model, y, z⁺, z⁻, args.numiter, args.γ)
                pred = 0.5f0*(z⁺[end] + z⁻[end])

                gs = gradient(ps) do
                    ls = contr_loss_network(model, args.batchsize, z⁺, z⁻, pred, y, args.γ)
                end
                grad_cosine_sim += [cosine_sim(gs[layer.Wₖ₋₁], gs_BP[layer.Wₖ₋₁]) for layer in model]
                Flux.Optimise.update!(args.opt, ps, gs) # update parameters
                ls_sum += ls

            elseif args.learnmode == "dualprop_betaL_lin_mse"
                # t=1: To begin with upstream activity is all zeroes so the first pass reduces to a standard forward pass.
                z⁺ = [x, Flux.activations(model, x)...]
                z⁻ = deepcopy(z⁺); 
                # nudge output units and update neurons in the top down direction
                # In this experimental setting \gamma is 1 for all layers except for the last one
                infer_zL_lin_mse!(model, y, z⁺, z⁻, args.γ)
                infer_zk_top_down!(model, z⁺, z⁻, 1.0f0)

                pred = 0.5f0*(z⁺[end] + z⁻[end])

                gs = gradient(ps) do
                    ls = contr_loss_network(model, args.batchsize, z⁺, z⁻, pred, y, args.γ)
                end
                grad_cosine_sim += [cosine_sim(gs[layer.Wₖ₋₁], gs_BP[layer.Wₖ₋₁]) for layer in model]
                Flux.Optimise.update!(args.opt, ps, gs) # update parameters
                ls_sum += ls

            elseif args.learnmode == "random_dualprop"
                # notice that we are dispatching to a different version of infer_z! here
                z⁺ *= 0f0
                z⁻ *= 0f0
                z⁺[1], z⁻[1] = x, x

                for t=1:args.numiter
                    k = rand(2:length(model)+1)
                    if k == length(model)+1
                        infer_zL!(model, y, z⁺, z⁻, args.γ)
                    else
                        model[k-1](z⁺[k-1], z⁺[k], z⁺[k+1], z⁻[k-1], z⁻[k], z⁻[k+1], args.γ)
                    end
                end

                pred = 0.5f0*(z⁺[end] + z⁻[end])

                gs = gradient(ps) do
                    ls = contr_loss_network(model, args.batchsize, z⁺, z⁻, pred, y, args.γ)
                end
                grad_cosine_sim += [cosine_sim(gs[layer.Wₖ₋₁], gs_BP[layer.Wₖ₋₁]) for layer in model]
                Flux.Optimise.update!(args.opt, ps, gs) # update parameters
                ls_sum += ls
            
            # multistep variant of dualpropagation performs a weight update after every args.numiter full passes bottom-->top-->bottom.
            elseif args.learnmode == "multistep_dualprop"
                z⁺, z⁻ = infer_z!(model, x, y, args.numiter, args.γ)
                pred = 0.5f0*(z⁺[end] + z⁻[end])
                gs = gradient(ps) do
                    ls = contr_loss_network(model, args.batchsize, z⁺, z⁻, pred, y, args.γ)
                end
                grad_cosine_sim += [cosine_sim(gs[layer.Wₖ₋₁], gs_BP[layer.Wₖ₋₁]) for layer in model]
                Flux.Optimise.update!(args.opt, ps, gs) # update parameters

                if args.multistep > 0
                    for s=2:args.multistep
                        z⁺, z⁻ = infer_z!(model, y, z⁺, z⁻, args.numiter, args.γ)
                        pred = 0.5f0*(z⁺[end] + z⁻[end])
                        gs = gradient(ps) do
                            ls = contr_loss_network(model, args.batchsize, z⁺, z⁻, pred, y, args.γ)
                        end
                        Flux.Optimise.update!(args.opt, ps, gs) # update parameters
                    end
                end
                ls_sum += ls
            
            # Dualprop with recurrent updates done "in parallel". 
            # The updates don't actually happen in parallel here, but by using deepcopies of the state vectors it is algorithmically equivalent to a parallel updates.
            elseif args.learnmode == "parallel_dualprop"
                # initialize neurons with zero activity and continue to update all units in parallel.
                z⁺ = 0f0*z⁺; z⁺[1] .= x
                z⁻ = deepcopy(z⁺)
                cp_z⁺ = deepcopy(z⁺)
                cp_z⁻ = deepcopy(z⁺)

                for t=1:args.numiter
                    for k=2:length(z⁺)-1
                        z⁺[k], z⁻[k] = model[k-1](cp_z⁺[k-1], z⁺[k], cp_z⁺[k+1], cp_z⁻[k-1], z⁻[k], cp_z⁻[k+1], args.γ)
                    end
                    # It takes T=length(model) parallel updates for a meaningful prediction to exist at the top layer.
                    # We do not want a FB signal from the loss function to the output units untill this is the case.
                    if t >= length(model)
                        z⁺[end] .= (model[end].Wₖ₋₁*(cp_z⁺[end-1] + cp_z⁻[end-1]) .+ 2*model[end].bₖ₋₁ .+ args.γ*y)/(2 + args.γ)
                        z⁻[end] .= (model[end].Wₖ₋₁*(cp_z⁺[end-1] + cp_z⁻[end-1]) .+ 2*model[end].bₖ₋₁ .- args.γ*y)/(2 - args.γ)
                    else
                        z⁺[end] .= model[end](z⁺[end-1])
                        z⁻[end] .= model[end](z⁻[end-1])
                    end
                    cp_z⁺ = deepcopy(z⁺)
                    cp_z⁻ = deepcopy(z⁻)
                end

                pred = 0.5f0*(z⁺[end] + z⁻[end])
                gs = gradient(ps) do
                    ls = contr_loss_network(model, args.batchsize, z⁺, z⁻, pred, y, args.γ)
                end
                Flux.Optimise.update!(args.opt, ps, gs) # update parameters
                ls_sum += ls

                grad_cosine_sim += [cosine_sim(gs[layer.Wₖ₋₁], gs_BP[layer.Wₖ₋₁]) for layer in model]
                Flux.Optimise.update!(args.opt, ps, gs) # update parameters

            else
                throw(ArgumentError("mode must be either \"dualprop\", \"multistep_dualprop\" or \"backprop\" or \"parallel_dualprop\""))
            end

        end
        metrics["train_contr_loss"][epoch] = ls_sum/batch_index
        for (i, g) in enumerate(grad_cosine_sim)
            metrics["grad_cosine_sim_$i"][epoch] = g/batch_index
        end

        metrics["runtime"][epoch] = convert(Float32, round3(time() - t1))
        @info "Epoch $epoch done in $(metrics["runtime"][epoch])s"
        ## Report on test data
        metrics["train_loss"][epoch], metrics["train_acc"][epoch] = evaluate(train_loader, model, device, "Train eval")
        metrics["val_loss"][epoch], metrics["val_acc"][epoch] = evaluate(val_loader, model, device, "Val eval")

        # Save snapshot of subset of filters
        if epoch==1 || epoch%5==0
            plot_filters(cpu(model[1].Wₖ₋₁), "$outpath/W0_epoch$epoch.png")
        end
        flush(io) # write all buffered messages to file
        
        # save best model
        if args.saveall == true
            m = cpu(model)
            @save "$(modelpath)/epoch$(epoch).bson" m
        end            
        if metrics["val_acc"][epoch] > acc_best
                m = cpu(model)
                acc_best = metrics["val_acc"][epoch]
                best_model_path = "$(modelpath)/epoch$(epoch).bson"
                @save best_model_path m
        end

    end # end of loop over epochs

    @load best_model_path m; m_best = device(m)
    println("best_model_path: ", best_model_path)
    test_loss, test_acc = evaluate(test_loader, m_best, device, "test eval")
    metrics["test_loss"], metrics["test_acc"] = [test_loss], [test_acc]

    @save "$(outpath)/metrics.bson" metrics
    plot_metrics(model, "$outpath/metrics.png", metrics)
    plot_grad_cosine_sim(model, "$outpath/grad_cossim.png", metrics)

    return cpu(model), metrics
end