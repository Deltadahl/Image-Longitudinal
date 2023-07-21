using Glob

# Specify how many lines to keep
number_of_rows_to_keep = 311  # change this as needed

# Function to process a file
function process_file(filename::String)
    # Read the file
    data = readlines(filename)

    # Check if the file has more lines than the specified number
    if length(data) > number_of_rows_to_keep
        # If it does, keep only the specified number of lines
        data = data[1:number_of_rows_to_keep]
    end

    # Overwrite the file with the truncated data
    open(filename, "w") do io
        for line in data
            write(io, line * "\n")
        end
    end
end

# Get a list of all .txt files in the directory
file_list = glob("*.txt", "saved_losses/try_31")

# Process all files
for filename in file_list
    process_file(filename)
end
