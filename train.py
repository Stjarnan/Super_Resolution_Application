from tools.nn import SRGAN
from tools.dataloader import BATCH
from keras.optimizers import Adam
import time
from imutils import paths
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# hyperparameters
EPOCHS = 5000
BATCH_SIZE = 8
INPUT_SHAPES = (32, 32, 3)
OUTPUT_SHAPES = (128, 128, 3)

# dataset 
dataset_path = '/dataset'

# load dataset
print("Loading dataset..")
image_paths = list(paths.list_images(dataset_path))

# only keep amount of images needed for learning
data = []
stored = 0
photos_needed = 2000

for image in image_paths:
    # check if enough data has been stored in 'data'
    # otherwise, store 'image' in 'data'
    if stored == photos_needed:
        break
    else:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)
        stored += 1

# Optimizer (for all nets)
OPT = Adam(0.0002, 0.5)

# Build and compile VGG
print("Compiling vgg..")
vgg = SRGAN.vgg(OUTPUT_SHAPES)
vgg.compile(loss="mse", optimizer=OPT, metrics=["accuracy"])

# build and compile the discriminator
print("Compiling discriminator..")
discriminator = SRGAN.discriminator(OUTPUT_SHAPES)
discriminator.trainable = False
discriminator.compile(loss="mse", optimizer=OPT, metrics=["accuracy"])

# build generator
print("Building the generator..")
generator = SRGAN.generator(INPUT_SHAPES)

# build and compile adversial model
print("Building and compiling the adversial model..")
model = SRGAN.build(INPUT_SHAPES, generator, discriminator, vgg)
model.compile(loss=["binary_crossentropy", "mse"],
    loss_weights=[1e-3, 1], optimizer=OPT)


# function to save images in a comparative way during training
def save_images(low_resolution_image, original_image, generated_image, path):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(low_resolution_image)
    ax.axis("off")
    ax.set_title("Low-resolution")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image)
    ax.axis("off")
    ax.set_title("Original")
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image)
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(path)

# Show epoch
for epoch in range(EPOCHS):
    print("Epoch:{}".format(epoch))

    # create batch
    high_res, low_res = BATCH.get_batch(data, BATCH_SIZE, (128, 128), INPUT_SHAPES)

    # normalize data
    # range should be [-1, 1]
    high_res = high_res / 127.5 - 1
    low_res = low_res / 127.5 - 1

    # run generator (create high res from low res)
    fake_HR = generator.predict(low_res)

    # real and fake labels
    real_labels = np.ones((BATCH_SIZE, 8, 8, 1))
    fake_labels = np.zeros((BATCH_SIZE, 8, 8, 1))

    # train discriminator on real and fake images
    discriminator_real = discriminator.train_on_batch(high_res, real_labels)
    discriminator_fake = discriminator.train_on_batch(fake_HR, fake_labels)

    # discriminator loss
    discriminator_loss = 0.5 * np.add(discriminator_real, discriminator_fake)
    print("Discriminator loss: {}".format(discriminator_loss))


    # extract features from high res images 
    features = vgg.predict(high_res)

    # train the generator
    generator_loss = model.train_on_batch([low_res, high_res], [real_labels, features])

    # generator loss
    print("Generator loss: {}".format(generator_loss))

    # Save images every 200 epochs
    if epoch % 200 == 0:

        generated_images = generator.predict_on_batch(low_res)

        for index, img in enumerate(generated_images):
            save_images(low_res[index], high_res[index], img, path="output/img_{}_{}".format(epoch, index))
        
    
# save models
generator.save_weights("generator.h5")
discriminator.save_weights("discriminator.h5")