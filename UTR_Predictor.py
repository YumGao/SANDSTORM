# Load packages
import sys
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")

import tensorflow as tf
import keras as tfk
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gc

import util
import GA_util
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from tensorflow.keras.optimizers import Adam

print('Modules loaded.')

# Model parameters
epoch_num = 1
iterations = 1
latent_dim = 128
learning_rate = 0.001
batch_size = 1024

# Load and filter input data
data = pd.read_csv('/home/nanoribo/NGSprocessing/CELL_IVT/feature/mRNA_halflife_highquaset_withfeatures.csv')
#data.to_csv('/home/nanoribo/NGSprocessing/CELL_IVT/feature/mRNA_halflife_highquaset_withfeatures_10.csv',index=False)

mask = ~data['sequence'].str.upper().str.contains('N', na=False)
data = data[mask].copy()
data = data[data['sequence'].apply(len) == 130].reset_index(drop=True)

# Feature and label extraction
seq_len = len(data['sequence'].iloc[0])
ppm_len = seq_len
y = data['half_life'].values

# Label transformation
stand = StandardScaler()
y_transformed = stand.fit_transform(y.reshape(-1, 1))

est = KBinsDiscretizer(n_bins=10, encode='ordinal')
encoded_vals = est.fit_transform(y.reshape(-1, 1))
print("Label bins:", encoded_vals.shape)

# One-hot encode sequences
utrs = util.one_hot_encode(data[['sequence']])
indices = np.arange(utrs.shape[0])

# Dataset generator
def create_dataset(x1, y, batch_size):
    def generator():
        for i in range(0, len(x1), batch_size):
            batch_x1 = x1[i:i+batch_size]
            batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
            batch_y = y[i:i+batch_size]
            yield (batch_x1, batch_x2), batch_y

    output_signature = (
        (
            tf.TensorSpec(shape=(batch_size, 4, seq_len), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, ppm_len, ppm_len), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

# Optional: memory cleanup
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


# 固定分割方式
utr_train, utr_test, y_train, y_test, _, _ = train_test_split(
        utrs, y_transformed, indices, test_size=0.2, stratify=encoded_vals,random_state=42)
    
np.save('utr_test.npy', utr_test)
np.save('y_test.npy',   y_test)   

    # Ensure divisible by batch size
num_samples = (len(utr_train) // batch_size) * batch_size
utr_train, y_train = utr_train[:num_samples], y_train[:num_samples]

    # Prepare data
train_dataset = create_dataset(utr_train, y_train, batch_size=batch_size)

# Training loop
for i in range(iterations):
    # Store results
    joint_mse_save, joint_r2_save, joint_spearman_save = [], [], []
    ppm_test = GA_util.prototype_ppms_fast(utr_test)

    # Model setup
    joint_model = GA_util.create_SANDSTORM(
        seq_len=seq_len,
        ppm_len=ppm_len,
        latent_dim=latent_dim,
        internal_activation='relu'
    )
    joint_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Train
    hist = joint_model.fit(
        train_dataset,
        epochs=epoch_num,
        validation_data=([utr_test, ppm_test], y_test),
        callbacks=[MemoryCleanupCallback()]
    )

    # Save model
    os.makedirs("./saved_model", exist_ok=True)
    joint_model.save("./saved_model/sandstorm_model_ori.h5")



