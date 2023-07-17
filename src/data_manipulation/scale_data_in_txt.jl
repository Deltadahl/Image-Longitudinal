using Glob

# Constant for multiplication
MULTIPLIER = 1/5

# Function to process a file
function process_file(filename::String)
    # Read the file
    data = readlines(filename)

    # Process the data
    processed_data = Float64[]
    for line in data
        num = parse(Float64, line)
        push!(processed_data, num * MULTIPLIER)
    end

    # Overwrite the file with the processed data
    open(filename, "w") do io
        for num in processed_data
            write(io, string(num) * "\n")
        end
    end
end

# Get a list of all .txt files in the directory
file_list = glob("*.txt", "saved_losses/try_17")

# Process all files
for filename in file_list
    process_file(filename)
end
