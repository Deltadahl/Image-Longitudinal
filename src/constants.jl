# constants.jl
using Flux


const CLASSES = Dict(
    1 => "CNV",
    2 => "DME",
    3 => "DRUSEN",
    4 => "NORMAL"
)
const BATCH_SIZE = 16
const DIRECTORY_PATH = "data/data_resized/all"
const DEVICE = gpu
const OUTPUT_IMAGE_DIR = "output_images"
