from tools.nn import SRGAN
from tools.dataloader import BATCH
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parse argument (to get image)
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help="image to process")
args = vars(parser.parse_args())

img = cv2.imread(args['image'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# build generator and discriminator
discriminator = SRGAN.discriminator((128, 128, 3))
generator = SRGAN.generator((32, 32, 3))

# load weights
generator.load_weights("generator.h5")

# processing
high_res, low_res = BATCH.get_batch([img], 1, (128, 128), (32, 32))
low_res = low_res / 127.5 - 1

# predict
generated = generator.predict_on_batch(low_res)

# save/return image
for img in enumerate(generated):
    img = cv2.normalize(img[1], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('generated.png', img)

