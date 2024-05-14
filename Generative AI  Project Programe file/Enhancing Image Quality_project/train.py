import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.python.ops.gen_experimental_dataset_ops import load_dataset

from model import generator, discriminator
import os
import numpy as np

# Set GPU memory growth to avoid allocation errors
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

# Define hyperparameters
batch_size = 16
epochs = 1000
lr = 1e-4
lambda_adv = 0.001  # Weight for adversarial loss
lambda_content = 0.01  # Weight for content loss

# Define loss functions
mse_loss = MeanSquaredError()
bce_loss = BinaryCrossentropy()

# Initialize generator and discriminator models
gen_model = generator()
disc_model = discriminator()

# Define optimizers
gen_optimizer = Adam(learning_rate=lr)
disc_optimizer = Adam(learning_rate=lr)

# Compile discriminator
disc_model.compile(loss=bce_loss, optimizer=disc_optimizer)

# Define combined model (GAN)
input_lr = tf.keras.layers.Input(shape=(None, None, 3))
output_hr = gen_model(input_lr)
output_validity = disc_model(output_hr)
combined_model = tf.keras.models.Model(input_lr, [output_hr, output_validity])

# Compile combined model
combined_model.compile(loss=[mse_loss, bce_loss], loss_weights=[lambda_content, lambda_adv], optimizer=gen_optimizer)

# Load dataset (Assuming you have a function load_dataset() to load your dataset)
train_dataset = load_dataset()

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i, batch in enumerate(train_dataset):
        input_lr_batch, target_hr_batch = batch

        # Train discriminator
        fake_hr_batch = gen_model.predict(input_lr_batch)
        disc_real_loss = disc_model.train_on_batch(target_hr_batch, np.ones(batch_size))
        disc_fake_loss = disc_model.train_on_batch(fake_hr_batch, np.zeros(batch_size))
        disc_loss = 0.5 * np.add(disc_real_loss, disc_fake_loss)

        # Train generator
        gen_loss = combined_model.train_on_batch(input_lr_batch, [target_hr_batch, np.ones(batch_size)])

        # Print progress
        print(f"Batch {i + 1}/{len(train_dataset)} | Disc Loss: {disc_loss}, Gen Loss: {gen_loss}")

    # Save weights every few epochs
    if (epoch + 1) % 10 == 0:
        gen_model.save_weights(f"generator_weights_epoch_{epoch + 1}.h5")
        disc_model.save_weights(f"discriminator_weights_epoch_{epoch + 1}.h5")
