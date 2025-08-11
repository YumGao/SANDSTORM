# UTR_Predictor.py
# Load packages
import sys
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")

import tensorflow as tf
import keras as tfk
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import pandas as pd# UTR_Predictor.py
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
# Model parameters
epoch_num = 1
iterations = 1
latent_dim = 128
learning_rate = 0.001
batch_size = 1248

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
    print("ppm_test",ppm_test)
    print("utr_test",utr_test)

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
    os.makedirs("./saved_model", exist_ok=Trimport sys
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")

import os
import gc
import time
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

import util
import GA_util

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def load_and_filter_data(cfg):
    data = pd.read_csv(cfg.data.input_csv)
    mask = ~data['sequence'].str.upper().str.contains('N', na=False)
    data = data[mask].copy()
    data = data[data['sequence'].apply(len) == 130].reset_index(drop=True)

    seq_len = len(data['sequence'].iloc[0])
    ppm_len = seq_len
    y = data['half_life'].values

    stand = StandardScaler()
    y_transformed = stand.fit_transform(y.reshape(-1, 1))

    est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    encoded_vals = est.fit_transform(y.reshape(-1, 1))

    utrs = util.one_hot_encode(data[['sequence']])
    indices = np.arange(utrs.shape[0])
    return utrs, y_transformed, encoded_vals, indices, seq_len, ppm_len

def create_dataset(x1, y, batch_size, seq_len, ppm_len):
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

@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)

    utrs, y_transformed, encoded_vals, indices, seq_len, ppm_len = load_and_filter_data(cfg)

    utr_train, utr_test, y_train, y_test, _, _ = train_test_split(
        utrs, y_transformed, indices, test_size=0.2, stratify=encoded_vals, random_state=42)

    np.save('utr_test.npy', utr_test)
    np.save('y_test.npy', y_test)

    num_samples = (len(utr_train) // cfg.train.batch_size) * cfg.train.batch_size
    utr_train, y_train = utr_train[:num_samples], y_train[:num_samples]

    train_dataset = create_dataset(utr_train, y_train, cfg.train.batch_size, seq_len, ppm_len)
    ppm_test = GA_util.prototype_ppms_fast(utr_test)

    resume_path = getattr(cfg.train, 'resume_path', None)
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        joint_model = load_model(resume_path)
    else:
        joint_model = GA_util.create_SANDSTORM(
            seq_len=seq_len,
            ppm_len=ppm_len,
            latent_dim=cfg.train.latent_dim,
            internal_activation=cfg.model.internal_activation
        )
        joint_model.compile(optimizer=Adam(learning_rate=cfg.train.learning_rate), loss='mse')

    os.makedirs(cfg.train.save_dir, exist_ok=True)

    start_time = time.time()
    hist = joint_model.fit(
        train_dataset,
        epochs=cfg.train.epoch_num,
        validation_data=([utr_test, ppm_test], y_test),
        callbacks=[MemoryCleanupCallback()]
    )
    elapsed = int(time.time() - start_time)
    final_loss = hist.history['loss'][-1]

    model_filename = f"sandstorm_loss{final_loss:.4f}_time{elapsed}s.h5"
    model_path = os.path.join(cfg.train.save_dir, model_filename)
    joint_model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
ue)
    joint_model.save("./saved_model/sandstorm_model.h5")





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
# Model parameters
epoch_num = 1
iterations = 1
latent_dim = 128
learning_rate = 0.001
batch_size = 1248

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
    print("ppm_test",ppm_test)
    print("utr_test",utr_test)

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
    joint_model.save("./saved_model/sandstorm_model.h5")




