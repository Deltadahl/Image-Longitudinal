# synthetic_utils_train.jl
function train!(model, x, y, opt, ps, loss_saver)
    batch_loss, back = Flux.pullback(() -> loss(model, x, y, loss_saver), ps)
    grads = back(1)
    Optimise.update!(opt, ps, grads)
    nothing
end

function save_losses(loss, filename)
    file = open(filename, "a")
    write(file, "$loss\n")
    close(file)
    nothing
end

function print_and_save(start_time, loss_saver, loss_saver_test, try_nr, save_nr)

    @show save_nr
    elapsed_time = time() - start_time
    hours, rem = divrem(elapsed_time, 3600)
    minutes, seconds = divrem(rem, 60)
    println("Time elapsed: $(floor(Int, hours))h $(floor(Int, minutes))m $(floor(Int, seconds))s")

    loss_train = loss_saver.loss / loss_saver.counter
    loss_test = loss_saver_test.loss / loss_saver_test.counter

    println("Loss train: $(Printf.@sprintf("%.9f", loss_train))")
    println("Loss test:  $(Printf.@sprintf("%.9f", loss_test))")

    folder_name = "synthetic_saved_losses/try_$(try_nr)/"
    if !isdir(folder_name)
        mkdir(folder_name)
    end

    save_losses(loss_train, folder_name * "loss_train.txt")
    save_losses(loss_test, folder_name * "loss_test.txt")

    nothing
end

function save_model(save_nr, model)
    model_copy = deepcopy(model)
    model_copy.to_random_effects = model_copy.to_random_effects |> cpu
    model_copy = model_copy |> cpu

    save_path = "synthetic_saved_models/save_nr_$(save_nr).jld2"
    save(save_path, "model", model_copy)
    println("saved model to $save_path")
    return nothing
end

function print_layers(model)
    for (i, layer) in enumerate(model)
        println("layer $i: ", repr(layer))
    end
    nothing
end

function print_model(model::SyntheticModel)
    println("Model Layers:")
    print_layers(model.to_random_effects.layers)
    nothing
end
