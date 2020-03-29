from tools.nn import SRGAN
from tools.dataloader import BATCH
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time
from imutils import paths
import cv2

# hyperparameters
EPOCHS = 500
BATCH_SIZE = 8
INPUT_SHAPES = (32, 32, 3)
OUTPUT_SHAPES = (128, 128, 3)

# dataset 
dataset_path = ''

# load dataset
print("Loading dataset..")
image_paths = list(paths.list_images(dataset_path))

# only keep amount of images needed for learning
data = []
stored = 0
photos_needed = 200000

for image in image_paths:

    # check if enough data has been stored in 'data'
    # otherwise, store 'image' in 'data'
    if stored == photos_needed:
        break
    else:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)

# Optimizer (for all nets)
OPT = Adam(0.0002, 0.5)

# Build and compile VGG
print("Compiling vgg..")
vgg = SRGAN.vgg(INPUT_SHAPES)
vgg.compile(loss="mse", optimizer=OPT, metrics=["accuracy"])

# build and compile the discriminator
print("Compiling discriminator..")
discriminator = SRGAN.discriminator(INPUT_SHAPES)
discriminator.compile(loss="mse", optimizer=OPT, metrics=["accuracy"])

# build generator
print("Building the generator..")
generator = SRGAN.generator(INPUT_SHAPES)

# build and compile adversial model
print("Building and compiling the adversial model..")
model = SRGAN.build(INPUT_SHAPES, generator, discriminator, vgg)
model.compile(loss=["binary_crossentropy", "mse"],
    loss_weights=[1e-3, 1], optimizer=OPT)

# Add visualization
tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
tensorboard.set_model(generator)
tensorboard.set_model(discriminator)

# Show epoch
for epoch in range(EPOCHS):
    print("Epoch:{}".format(epoch))

    

