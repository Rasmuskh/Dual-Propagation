struct dualprop_Dense{F1, M1<:AbstractMatrix, M2, V1, V2}
    Wₖ₋₁::M1
    Wₖ::M2
    bₖ₋₁::V1
    bₖ::V2
    σ::F1 # activation function
    function dualprop_Dense(Wₖ₋₁::M1, Wₖ::M2, bₖ₋₁::V1, bₖ::V2, σ::F1 = identity) where {M1<:AbstractMatrix, M2, V1, V2, F1}
        new{F1, M1, M2, V1, V2}(Wₖ₋₁, Wₖ, bₖ₋₁, bₖ, σ)
    end
end

Flux.@functor dualprop_Dense

# pretty printing stuff 🌈
function Base.show(io::IO, l::dualprop_Dense)
    print(io, "dualprop_Dense(size(Wₖ₋₁)=", size(l.Wₖ₋₁))
    l.Wₖ == false || print(io, ", size(Wₖ)=", size(l.Wₖ))
    l.bₖ₋₁ == false || print(io, ", size(bₖ₋₁)=", size(l.bₖ₋₁))
    l.bₖ == false || print(io, ", size(bₖ)=", size(l.bₖ))
    l.σ == identity || print(io, ", σ=", l.σ)
    print(io, ")")
end

Flux.trainable(a::dualprop_Dense) = (a.Wₖ₋₁, a.Wₖ, a.bₖ₋₁, a.bₖ)

# Feed-forward inference pass. The recurrent pass below is equivalent to this FF pass when (z⁺ₖ₊₁, z⁻ₖ₊₁) are both zero=#
function (a::dualprop_Dense)(x::AbstractVecOrMat)
    return a.σ.(a.Wₖ₋₁*x .+ a.bₖ₋₁)
end

(a::dualprop_Dense)(x::AbstractArray) = reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

# Recurrent inference pass. Infer latent state (z⁻ₖ, z⁺ₖ) given neighbouring layers activations (z⁻ₖ₋₁, z⁺ₖ₋₁) and (z⁺ₖ₊₁, z⁺ₖ₊₁)
function (a::dualprop_Dense)(z⁺ₖ₋₁::AbstractVecOrMat, z⁺ₖ::AbstractVecOrMat, z⁺ₖ₊₁::AbstractVecOrMat, z⁻ₖ₋₁::AbstractVecOrMat, z⁻ₖ::AbstractVecOrMat, z⁻ₖ₊₁::AbstractVecOrMat, γ)
    FF = a.Wₖ₋₁ * (z⁺ₖ₋₁ + z⁻ₖ₋₁) .+ 2f0 * a.bₖ₋₁
    FB = γ*transpose(a.Wₖ)*(z⁺ₖ₊₁ - z⁻ₖ₊₁)
    z⁺ₖ .= a.σ.(0.5f0 * (FF + FB))
    z⁻ₖ .= a.σ.(0.5f0 * (FF - FB))
    return z⁺ₖ, z⁻ₖ
end

# Compute energy of dualprop_Dense layer given activations zₖ₋₁ and zₖ
function energy(l::dualprop_Dense, zₖ, zₖ₋₁)
    batchsize = size(zₖ₋₁, 2)
    return 0.5f0*sum(abs2, l.Wₖ₋₁*zₖ₋₁  .+ l.bₖ₋₁ .- zₖ)/batchsize
end

function contr_loss_layer(l::dualprop_Dense, z⁺ₖ::AbstractVecOrMat, z⁺ₖ₋₁::AbstractVecOrMat, z⁻ₖ::AbstractVecOrMat, z⁻ₖ₋₁::AbstractVecOrMat)
    return energy(l, z⁺ₖ, z⁺ₖ₋₁) + energy(l, z⁺ₖ, z⁻ₖ₋₁) - energy(l, z⁻ₖ, z⁺ₖ₋₁) - energy(l, z⁻ₖ, z⁻ₖ₋₁)
end

function contr_loss_network(model, batchsize, z⁺, z⁻, pred, y, γ)
    ls = 0f0
    for i=1:length(model)
        ls += γ*contr_loss_layer(model[i], z⁺[i+1], z⁺[i], z⁻[i+1], z⁻[i])
    end

    # ls += sum(abs2, 0.5f0*model[end].Wₖ₋₁*(z⁺[end-1] + z⁻[end-1])- y)/batchsize
    # ls += sum(diag(transpose(z⁺[end] - z⁻[end])*(0.5f0*(z⁺[end] + z⁻[end]) - y)))
    # ls += sum(diag(transpose(z⁺[end] - z⁻[end])*(z⁺[end] - z⁻[end])))
    Δ = pred - y
    ls += sum(diag(transpose(2*γ*Δ)*Δ))/batchsize

    return ls
end
    
function build_dualprop_MLP(N=[784, 1024, 512, 256, 10])
    W = [Flux.glorot_uniform(N[k+1], N[k]) for k=1:length(N)-1]
    b = [zeros(Float32, N[k+1]) for k=1:length(N)-1]
    hiddenlayers = [dualprop_Dense(W[k], W[k+1], b[k], b[k+1], relu) for k=1:length(N)-2]
    outputlayer = dualprop_Dense(W[end], false, b[end], false, identity)
    model = Chain(hiddenlayers..., outputlayer)
    return model
end    
