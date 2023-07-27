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

#Define the encoder
function create_synthetic_model()
    model_base = ResNet(18; inchannels = 1, nclasses = 3)
    # model_base = ResNet(18; inchannels = 1, nclasses = OUTPUT_SIZE_ENCODER)
    # model = Chain(model_base, relu, Dense(OUTPUT_SIZE_ENCODER, 3))
    return model_base
    #Flatten ,then one fully connected layer
    # model = Chain(Flux.flatten, Dense(224*224, 3))
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

    # mse92 = sum(mean((y_approx[1,:] .- y[1,:]).^2, dims=1))
    # mse111 = sum(mean((y_approx[2,:] .- y[2,:]).^2, dims=1))
    # mse50 = sum(mean((y_approx[3,:] .- y[3,:]).^2, dims=1))

    # mse92 = sum((x_batch_test[92, :] .- y_approx[1,:]).^2)
    # mse111 = sum((x_batch_test[111, :] .- y_approx[2,:]).^2)
    # mse50 = sum((x_batch_test[50, :] .- y_approx[3,:]).^2)

    # mse92 = sum(mean((x_batch_test[92, :] .- y[1,:]).^2, dims=1))
    # mse111 = sum(mean((x_batch_test[111, :] .- y[2,:]).^2, dims=1))
    # mse50 = sum(mean((x_batch_test[50, :] .- y[3,:]).^2, dims=1))

    loss_mse = mean(mse92 + mse111 + mse50)
    # loss_mse = sum(mean((y .- y_approx).^2, dims=1))

    update_loss!(loss_saver, loss_mse, batch_size)
    return loss_mse
end

Flux.trainable(m::SyntheticModel) = (m.to_random_effects,)
