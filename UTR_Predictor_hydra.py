# train_sandstorm_hydra.py
import sys
import os
import gc
import hydra
from omegaconf import DictConfig
import tensorflow as tf
import keras as tfk
tfkl = tf.keras.layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from tensorflow.keras.optimizers import Adam

# Add custom module path
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")
import util
import GA_util

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU按需增长，防止显存被全部占用
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU(s): {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"Error setting up GPU memory growth: {e}")
    else:
        print("No GPU detected, running on CPU.")


class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    """Free memory at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def create_dataset(x1, y, batch_size, seq_len, ppm_len):
    """Create tf.data.Dataset for SANDSTORM."""
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


@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    setup_gpu()

    print("Hydra configuration loaded:")
    print(cfg)

    # Load and filter input data
    data = pd.read_csv(cfg.data.input_csv)
    mask = ~data['sequence'].str.upper().str.contains('N', na=False)
    data = data[mask].copy()
    data = data[data['sequence'].apply(len) == 130].reset_index(drop=True)

    # Feature/label extraction
    seq_len = len(data['sequence'].iloc[0])
    ppm_len = seq_len
    y = data['half_life'].values

    # Label transformation
    stand = StandardScaler()
    y_transformed = stand.fit_transform(y.reshape(-1, 1))
    est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    encoded_vals = est.fit_transform(y.reshape(-1, 1))

    # One-hot encode sequences
    utrs = util.one_hot_encode(data[['sequence']])
    indices = np.arange(utrs.shape[0])

    # Train/test split
    utr_train, utr_test, y_train, y_test, _, _ = train_test_split(
        utrs, y_transformed, indices,
        test_size=0.2, stratify=encoded_vals, random_state=42
    )

    # Save test sets
    np.save('utr_test.npy', utr_test)
    np.save('y_test.npy', y_test)

    # Ensure divisible by batch size
    num_samples = (len(utr_train) // cfg.train.batch_size) * cfg.train.batch_size
    utr_train, y_train = utr_train[:num_samples], y_train[:num_samples]

    # Prepare training dataset
    train_dataset = create_dataset(
        utr_train, y_train,
        batch_size=cfg.train.batch_size,
        seq_len=seq_len,
        ppm_len=ppm_len
    )

    # Training iterations
    for i in range(cfg.train.iterations):
        print(f"=== Iteration {i+1}/{cfg.train.iterations} ===")

        # Precompute PPM for test set
        ppm_test = GA_util.prototype_ppms_fast(utr_test)

        # Create model
        joint_model = GA_util.create_SANDSTORM(
            seq_len=seq_len,
            ppm_len=ppm_len,
            latent_dim=cfg.train.latent_dim,
            internal_activation=cfg.model.internal_activation
        )
        joint_model.compile(
            optimizer=Adam(learning_rate=cfg.train.learning_rate),
            loss='mse'
        )

        # Train model
        hist = joint_model.fit(
            train_dataset,
            epochs=cfg.train.epoch_num,
            validation_data=([utr_test, ppm_test], y_test),
            callbacks=[MemoryCleanupCallback()]
        )

        # Save model
        os.makedirs(cfg.train.save_dir, exist_ok=True)
        model_path = os.path.join(cfg.train.save_dir, f"{cfg.model_name}_model.h5")
        joint_model.save(model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
