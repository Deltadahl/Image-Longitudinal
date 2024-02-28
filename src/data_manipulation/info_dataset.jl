using Images
using FileIO
using Glob
using Printf
using DataStructures

function get_image_data(path::String, extension::String)
    # Get a list of all .jpeg files in the directory
    files = glob("*.$extension", path)

    # Initialize dictionaries to store the image sizes, counts, and unique patients
    image_sizes = DefaultDict{Tuple{Int64,Int64},Int64}(0)
    patients = DefaultDict{String, Set{String}}(Set{String})
    class_counts = DefaultDict{String, Int64}(0)

    # Loop over the files
    for file in files
        # Load the image
        image = load(file)

        # Get the size of the image and increment its count in the dictionary
        image_sizes[size(image)] += 1

        # Get the patient and class from the filename
        base_name = basename(file)
        class, patient, _ = split(base_name, "-")

        # Add the patient to the set of unique patients for this class
        push!(patients[class], patient)

        # Increment the count for this class
        class_counts[class] += 1
    end

    return image_sizes, patients, class_counts
end

function write_to_file(filename::String, data)
    open(filename, "w") do io
        for line in data
            write(io, line)
        end
    end
end

function main()
    # Call the functions
    base_path = "data/CellData/OCT"
    subfolders = [
        "train/NORMAL",
        "train/DRUSEN",
        "train/DME",
        "train/CNV",
        "test/NORMAL",
        "test/DRUSEN",
        "test/DME",
        "test/CNV",
    ]
    total_image_sizes = DefaultDict{Tuple{Int64,Int64},Int64}(0)
    total_patients = Set{String}()
    total_class_counts = DefaultDict{String, Int64}(0)
    results = []

    for subfolder in subfolders
        path_current = joinpath(base_path, subfolder)
        image_sizes, patients, class_counts = get_image_data(path_current, "jpeg")

        push!(results, "Statistics for $subfolder\n")
        for (size, count) in sort(collect(image_sizes), by = x -> (x[1][2], x[1][1]))
            push!(results, "Image size: $(size[1]) x $(size[2]), count: $count\n")
        end
        for (class, patient_set) in patients
            push!(results, "Number of unique patients in $class: $(length(patient_set))\n")
            union!(total_patients, patient_set)
        end
        for (class, count) in class_counts
            push!(results, "Number of occurrences of $class: $count\n")
            total_class_counts[class] += count
        end
        push!(results, "\n")

        for (size, count) in image_sizes
            total_image_sizes[size] += count
        end
    end

    push!(results, "Total Statistics\n")
    for (size, count) in sort(collect(total_image_sizes), by = x -> (x[1][2], x[1][1]))
        push!(results, "Image size: $(size[1]) x $(size[2]), count: $count\n")
    end
    for (class, count) in total_class_counts
        push!(results, "Total number of occurrences of $class: $count\n")
    end
    push!(results, "Total number of unique patients: $(length(total_patients))\n")

    write_to_file("dataset_statistics.txt", results)
end

@time main()
