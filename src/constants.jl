# constants.jl
using Flux

const CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
const BATCH_SIZE = 8
const DIRECTORY_PATH = "data/data_resized/all"
const DEVICE = gpu
