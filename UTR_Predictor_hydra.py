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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
import random


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


def create_dataset(x1, y, max_len, batch_size):
    """Create tf.data.Dataset for SANDSTORM."""
    def generator():
        for i in range(0, len(x1), batch_size):
            batch_x1 = x1[i:i+batch_size]
            batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
            batch_y = y[i:i+batch_size]
            yield (batch_x1, batch_x2), batch_y

    output_signature = (
        (
            tf.TensorSpec(shape=(batch_size, 4, max_len), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, max_len, max_len), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

def create_val_dataset(x1, y, max_len, batch_size):
    def generator():
        for i in range(0, len(x1), batch_size):
            batch_x1 = x1[i:i+batch_size]
            batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
            batch_y = y[i:i+batch_size]
            yield (batch_x1, batch_x2), batch_y

    output_signature = (
        (
            tf.TensorSpec(shape=(None, 4, max_len), dtype=tf.float32),  # None for batch dim
            tf.TensorSpec(shape=(None, max_len, max_len), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

# Then in main(), replace validation_data with:


def save_split_dfs(trainval_df, test_df, base_dir, trainval_test_ratio, max_len):
    split_dir = os.path.join(base_dir, f"trainval{1-trainval_test_ratio}_test{trainval_test_ratio}_len{max_len}")
    os.makedirs(split_dir, exist_ok=True)

    trainval_path = os.path.join(split_dir, "trainval.csv")
    test_path = os.path.join(split_dir, "test.csv")

    trainval_df.to_csv(trainval_path, index=False)
    test_df.to_csv(test_path, index=False)

    return split_dir, trainval_path, test_path


def load_and_filter_data(cfg):
    df = pd.read_csv(cfg.data.input_csv)
    mask = ~df['sequence'].str.upper().str.contains('N', na=False)
    df = df[mask].copy()
    df = df[df['sequence'].apply(len) == 201].reset_index(drop=True)
    max_len = df["sequence"].str.len().max()

    # Split train_val and test by ratio from config
    #trainval_ratio = 1 - cfg.data.holdout_test_ratio
    trainval_df, test_df = train_test_split(df, test_size=cfg.data.holdout_test_ratio, random_state=42)
    print(f"Spliting holdout test set: \n\ttrain & val ({1 - cfg.data.holdout_test_ratio},{trainval_df.shape})\n\ttest ({cfg.data.holdout_test_ratio},{test_df.shape})")

    # Save splits
    split_dir, trainval_path, test_path = save_split_dfs(trainval_df, test_df, cfg.data.split_dir, cfg.data.holdout_test_ratio, max_len)

    print(f"Saved trainval to {trainval_path}")
    print(f"Saved test to {test_path}")

    return trainval_df, max_len



def prepare_features_and_split(df, max_len, batch_size,val_size):
    """Transform labels, encode sequences, split dataset, and save test sets."""
    y = df['half_life'].values

    # Label transformation
    stand = StandardScaler()
    y_transformed = stand.fit_transform(y.reshape(-1, 1))
    est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    encoded_vals = est.fit_transform(y.reshape(-1, 1))

    # One-hot encode sequences
    utrs = util.one_hot_encode(df[['sequence']])
    indices = np.arange(utrs.shape[0])
    holdout_test_ratio: 0.05


    # Train/test split

    utr_train, utr_val, y_train, y_val, _, _ = train_test_split(
        utrs,
        y_transformed,
        indices,
        test_size=val_size,    
        stratify=encoded_vals,
        random_state=42
    )

    # Save test sets
    #np.save('utr_test.npy', utr_test)
    #np.save('y_test.npy', y_test)

    # Ensure divisible by batch size
    num_samples = (len(utr_train) // batch_size) * batch_size
    utr_train, y_train = utr_train[:num_samples], y_train[:num_samples]

    return utr_train, utr_val, y_train, y_val

@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    setup_gpu()

    print("Hydra configuration loaded:")
    print(cfg)

    filtered_df, max_len = load_and_filter_data(cfg)
    val_size = cfg.data.val_ratio/(1 - cfg.data.holdout_test_ratio)
    seq_train, seq_val, y_train, y_val = prepare_features_and_split(filtered_df, max_len, cfg.train.batch_size,val_size=val_size)

    # Prepare training dataset
    train_dataset = create_dataset(
        seq_train, y_train,
        max_len=max_len,
        batch_size=cfg.train.batch_size
        
    )
    val_dataset = create_val_dataset(
        seq_val, y_val, 
        max_len=max_len, 
        batch_size=cfg.train.batch_size)



    # Training iterations
    print(f"=== Start Training ===")


        # Create model
    joint_model = GA_util.create_SANDSTORM(
            seq_len=max_len,
            ppm_len=max_len,
            latent_dim=cfg.train.latent_dim,
            internal_activation=cfg.settings.model.internal_activation
        )

        # AdamW optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(
        learning_rate=cfg.train.learning_rate,  # e.g., 1e-3
        weight_decay=cfg.train.weight_decay, # small decay to stabilize training
        clipnorm=1.0
    )

    joint_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']  # optional: track mean absolute error
    )

        # Train model
    hist = joint_model.fit(
            train_dataset,
            epochs=cfg.train.epoch_num,
            validation_data=(val_dataset),
            callbacks = [
                MemoryCleanupCallback(),
                EarlyStopping(
                    monitor=cfg.train.early_stop_monitor,
                    patience=cfg.train.early_stop_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor=cfg.train.early_stop_monitor,
                    factor=cfg.train.reduce_lr_factor,
                    patience=cfg.train.reduce_lr_patience,
                    min_lr=cfg.train.min_lr,
                    verbose=1
                )
            ]
        )



        # Save model
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    model_path = os.path.join(cfg.train.save_dir, f"{cfg.model_name}_model.h5")
    joint_model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
