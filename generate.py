from tools.nn import SRGAN
from tools.dataloader import BATCH
import argparse
import cv2

# Parse argument (to get image)
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help="image to process")
args = parser.parse_args()

# build generator and discriminator
discriminator = SRGAN.discriminator((128, 128))
generator = SRGAN.generator((32, 32))

# load weights
generator.load_weights("generator.h5")

# processing
high_res, low_res = BATCH.get_batch([args['image']], 1, (128, 128), (32, 32))
high_res = high_res / 127.5 - 1
low_res = low_res / 127.5 - 1

# predict
generated = generator.predict_on_batch(low_res)

# save/return image
cv2.imwrite('generated.jpg', generated)
