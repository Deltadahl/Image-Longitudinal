using Distributed

# Add as many worker processes as there are CPU cores
addprocs(16)

@everywhere using Images
@everywhere using FileIO
@everywhere using ImageMorphology
@everywhere using Statistics
@everywhere using Glob

# Place your functions check_vertical_path, process_image here with @everywhere prefix.
@everywhere function check_vertical_path(mask, i, j)
    up_obsticle = false
    down_obsticle = false
    # Scan upwards
    for k in range(i, stop=1, step=-1)
        if !mask[k, j]
            up_obsticle = true
            break
        end
    end
    # Scan downwards
    for k in range(i, stop=size(mask, 1))
        if !mask[k, j]
            down_obsticle = true
            break
        end
    end
    return !(up_obsticle && down_obsticle)
end

@everywhere function process_image(img)
    img = Gray.(img)
    img_array = float32.(img)
    thresholded = img_array .> 0.75
    labeled = label_components(thresholded)
    corner_masks = []
    corners = [(1, 1), (1, size(img,2)), (size(img,1), 1), (size(img,1), size(img,2))]
    for corner in corners
        label = labeled[corner...]
        if label > 0
            mask = labeled .== label
            push!(corner_masks, mask)
        end
    end
    final_mask = if isempty(corner_masks)
        zeros(Bool, size(thresholded))
    else
        reduce(.|, corner_masks)
    end

    # Create a copy of the final_mask to avoid altering it during iteration
    new_mask = copy(final_mask)

    # check for mask pixels that cannot go straight up or down
    for j in 1:size(final_mask, 2)
        for i in 1:size(final_mask, 1)
            if final_mask[i, j] && !check_vertical_path(final_mask, i, j)
                new_mask[i, j] = false
            end
        end
    end

    # create a final expanded mask
    expanded_mask = copy(new_mask)
    n = 2
    for j in 1:size(new_mask, 2)
        for i in 1:size(new_mask, 1)
            if new_mask[i, j]
                # expand the mask in all directions by n pixels
                expanded_mask[max(i-n, 1):min(i+n, end), max(j-n, 1):min(j+n, end)] .= true
            end
        end
    end

    return expanded_mask, img_array
end

@everywhere function process_and_save_file(file, base_path_modified, subfolder)
    img_path = file
    img = load(img_path)
    mask, img_array = process_image(img)
    modified = copy(img_array)
    not_masked_pixels = sort(vec(img_array[.!mask]))
    darkest_pixels_avg = maximum(not_masked_pixels[1:div(end, 2)])
    # darkest_pixels_avg = median(vec(img_array[.!mask]))
    modified[mask] .= darkest_pixels_avg
    save_path = joinpath(base_path_modified, subfolder, basename(file))
    mkpath(dirname(save_path))
    save(save_path, Gray.(modified))
end

function main()
    base_path = "data/CellData/OCT"
    base_path_modified = "data/CellData/OCT_white_to_black"
    subfolders = [
        "test/NORMAL",
        "test/DRUSEN",
        "test/CNV",
        "test/DME",
        "train/NORMAL",
        "train/DRUSEN",
        "train/DME",
        "train/CNV",
    ]
    for subfolder in subfolders
        println("Processing $subfolder")
        directory = joinpath(base_path, subfolder)
        directory_modified = joinpath(base_path_modified, subfolder)
        files = glob("*.jpeg", directory)

        # Use @distributed to perform the file processing in parallel.
        # Note that we've modified the loop to use `pmap` function which is better for IO bounded operations.
        # The function to be applied to each file and the collection of files are passed as arguments to `pmap`.
        @time pmap(file -> process_and_save_file(file, base_path_modified, subfolder), files)
    end
end

main()
