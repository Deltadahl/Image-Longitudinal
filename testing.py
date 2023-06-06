import os
import numpy as np
import struct
from PIL import Image

# Load everything in some numpy arrays
with open('data/train-labels.idx1-ubyte', 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)

with open('data/train-images.idx3-ubyte', 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

# Make sure the images directory exists
if not os.path.exists('images'):
    os.makedirs('images')

# Save images
for i in range(len(labels)):
    image = Image.fromarray(images[i].reshape(28,28).astype(np.uint8))
    image.save(f'images/{labels[i]}_image_{i}.png')
