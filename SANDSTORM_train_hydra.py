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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
            # Set GPU memory growth to prevent full allocation
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

class DatasetManager:
    """Manages DataFrame loading, filtering, splitting, and dataset creation."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_len = None
        self.trainval_df = None
        self.test_df = None
        self.train_df = None
        self.val_df = None
        self.seq_train = None
        self.seq_val = None
        self.seq_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
    
    def load_and_filter_data(self):
        """Load and filter the input DataFrame."""
        df = pd.read_csv(self.cfg.data.input_csv)
        mask = ~df['sequence'].str.upper().str.contains('N', na=False)
        df = df[mask].copy()
        df = df[df['sequence'].apply(len) == 201].reset_index(drop=True)
        self.max_len = df["sequence"].str.len().max()
        
        # Split trainval and test
        self.trainval_df, self.test_df = train_test_split(
            df, 
            test_size=self.cfg.data.holdout_test_ratio, 
            random_state=42
        )
        print(f"Splitting holdout test set: \n\ttrain & val ({self.trainval_df.shape})\n\ttest ({self.test_df.shape})")
        
        # Save splits
        self._save_split_dfs()
        return self.trainval_df, self.max_len
    
    def _save_split_dfs(self):
        """Save trainval and test DataFrames to disk."""
        split_dir = os.path.join(self.cfg.data.split_dir, f"trainval{1-self.cfg.data.holdout_test_ratio}_test{self.cfg.data.holdout_test_ratio}")
        os.makedirs(split_dir, exist_ok=True)
        trainval_path = os.path.join(split_dir, "trainval.csv")
        test_path = os.path.join(split_dir, "test.csv")
        self.trainval_df.to_csv(trainval_path, index=False)
        self.test_df.to_csv(test_path, index=False)
        print(f"Saved trainval to {trainval_path}")
        print(f"Saved test to {test_path}")
        return split_dir, trainval_path, test_path
    
    def prepare_features_and_split(self, batch_size, val_size):
        """Split trainval into train/val randomly, encode sequences, fit scaler on train, transform val/test."""
        # Split trainval into train and val
        self.train_df, self.val_df = train_test_split(
            self.trainval_df,
            test_size=val_size,
            random_state=42
        )
        print(f"Splitting train/val: \n\ttrain ({self.train_df.shape})\n\tval ({self.val_df.shape})")
        
        # One-hot encode sequences for train, val, and test
        self.seq_train = util.one_hot_encode(self.train_df[['sequence']])
        self.seq_val = util.one_hot_encode(self.val_df[['sequence']])
        self.seq_test = util.one_hot_encode(self.test_df[['sequence']])
        
        # Get raw labels
        y_raw_train = self.train_df['half_life'].values
        y_raw_val = self.val_df['half_life'].values
        y_raw_test = self.test_df['half_life'].values
        
        # Fit scaler on train labels only
        self.scaler.fit(y_raw_train.reshape(-1, 1))
        
        # Transform train, val, and test labels using the fitted scaler
        self.y_train = self.scaler.transform(y_raw_train.reshape(-1, 1))
        self.y_val = self.scaler.transform(y_raw_val.reshape(-1, 1))
        self.y_test = self.scaler.transform(y_raw_test.reshape(-1, 1))
        
        # Ensure train is divisible by batch size
        num_samples = (len(self.seq_train) // batch_size) * batch_size
        self.seq_train, self.y_train = self.seq_train[:num_samples], self.y_train[:num_samples]
        
        return self.seq_train, self.seq_val, self.y_train, self.y_val
    
    def create_dataset(self, x1, y, batch_size, is_train=True):
        """Create tf.data.Dataset for training, validation, or test."""
        def generator():
            for i in range(0, len(x1), batch_size):
                batch_x1 = x1[i:i+batch_size]
                batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
                batch_y = y[i:i+batch_size]
                yield (batch_x1, batch_x2), batch_y
        
        batch_shape = batch_size if is_train else None
        output_signature = (
            (
                tf.TensorSpec(shape=(batch_shape, 4, self.max_len), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_shape, self.max_len, self.max_len), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(batch_shape, 1), dtype=tf.float32)
        )
        return tf.data.Dataset.from_generator(
            generator, 
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)
    
    def get_train_dataset(self, batch_size):
        """Return training dataset."""
        return self.create_dataset(self.seq_train, self.y_train, batch_size, is_train=True)
    
    def get_val_dataset(self, batch_size):
        """Return validation dataset."""
        return self.create_dataset(self.seq_val, self.y_val, batch_size, is_train=False)
    
    def get_test_dataset(self, batch_size):
        """Return test dataset."""
        return self.create_dataset(self.seq_test, self.y_test, batch_size, is_train=False)

@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    setup_gpu()
    print("Hydra configuration loaded:")
    print(cfg)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(cfg)
    filtered_df, max_len = dataset_manager.load_and_filter_data()
    val_size = cfg.data.val_ratio / (1 - cfg.data.holdout_test_ratio)
    seq_train, seq_val, y_train, y_val = dataset_manager.prepare_features_and_split(
        batch_size=cfg.train.batch_size, 
        val_size=val_size
    )
    
    # Prepare datasets
    train_dataset = dataset_manager.get_train_dataset(cfg.train.batch_size)
    val_dataset = dataset_manager.get_val_dataset(cfg.train.batch_size)
    
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
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        clipnorm=1.0
    )
    
    joint_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    hist = joint_model.fit(
        train_dataset,
        epochs=cfg.train.epoch_num,
        validation_data=val_dataset,
        callbacks=[
            MemoryCleanupCallback(),
            EarlyStopping(
                monitor=cfg.train.early_stop_monitor,
                patience=cfg.train.early_stop_patience,
                min_delta=cfg.train.early_stop_min_delta,
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
    final_epoch = len(hist.history['loss'])
    train_loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    model_path = os.path.join(
        cfg.train.save_dir, 
        f"{cfg.model_name}_e{final_epoch:03d}_tl{train_loss:.3f}_vl{val_loss:.3f}.h5"
    )
    joint_model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()