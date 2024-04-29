# synthetic_model.jl
using Flux
using Flux: Chain
using CUDA
using Statistics
using Zygote
using Metalhead
using Colors
using NNlib
CUDA.math_mode!(CUDA.PEDANTIC_MATH)
include("constants.jl")

function create_synthetic_model()
    model_base = ResNet(18; inchannels = 1, nclasses = 3)
    return model_base
end

mutable struct SyntheticModel
    to_random_effects
end

function (m::SyntheticModel)(x)
    return m.to_random_effects(x)
end

mutable struct LossSaverSynthetic
    loss::Float32
    counter::Float32
end

Zygote.@nograd function update_loss!(loss_saver::LossSaverSynthetic,  loss_mse, batch_size)
    loss_saver.loss += loss_mse
    loss_saver.counter += 1.0f0 * batch_size
end


function loss(m::SyntheticModel, x, y, loss_saver::LossSaverSynthetic)
    y_approx = m(x)
    batch_size = size(y, 2)

    loss_mse = sum(mean((y .- y_approx).^2, dims=1))

    update_loss!(loss_saver, loss_mse, batch_size)
    return loss_mse
end

Flux.trainable(m::SyntheticModel) = (m.to_random_effects,)
