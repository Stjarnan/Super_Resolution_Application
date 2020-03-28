from tools.nn import SRGAN
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# dataset 
dataset = ''

# path for the output model
output_path = "../../../../output/model.pkl"


# hyperparameters
EPOCHS = 500
BATCH_SIZE = 8
INPUT_SHAPES = (32, 32, 3)
OUTPUT_SHAPES = (128, 128, 3)

# Optimizer (for all nets)
OPT = Adam(0.0002, 0.5)

# Build and compile VGG
vgg = SRGAN.vgg(INPUT_SHAPES)
vgg.compile(loss="mse", optimizer=OPT, metrics=["accuracy"])

# build and compile the discriminator
discriminator = SRGAN.discriminator(INPUT_SHAPES)
discriminator.compile(loss="mse", optimizer=OPT, metrics=["accuracy"])

# build generator
generator = SRGAN.generator(INPUT_SHAPES)

# build and compile adversial model
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

