# constants.jl
using Flux

const BATCH_SIZE = 8
const DEVICE = gpu
const OUTPUT_IMAGE_DIR = "output_images"
const LATENT_DIM = 16 # TODO make larger.
const OUTPUT_SIZE_ENCODER = 1000
