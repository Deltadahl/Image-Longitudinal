using Glob

# specify the number of rows to keep
number_of_rows_to_keep = 1277   # replace with the actual number of rows

# directory of text files
directory = "saved_losses\\try_15\\"

# find all .txt files in the directory
files = glob("*.txt", directory)

# iterate over each file
for file in files
    # read all lines of the file
    lines = readlines(file)

    # check if the number of lines is more than number_of_rows_to_keep
    if length(lines) > number_of_rows_to_keep
        # truncate the file to only the first number_of_rows_to_keep lines
        lines = lines[1:number_of_rows_to_keep]

        # open the file for writing
        open(file, "w") do io
            # write the truncated lines back into the file
            for line in lines
                println(io, line)
            end
        end
    end
end

println("Done!")
