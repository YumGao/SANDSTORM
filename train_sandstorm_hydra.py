import sys
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

class DataProcessor:
    def __init__(self, seq_len=130, seq_col='sequence', label_col='half_life'):
        self.seq_len = seq_len
        self.seq_col = seq_col
        self.label_col = label_col
        self.scaler = StandardScaler()
        self.binner = KBinsDiscretizer(n_bins=10, encode='ordinal')

    def filter_data(self, csv_path):
        data = pd.read_csv(csv_path)
        mask = ~data[self.seq_col].str.upper().str.contains('N', na=False)
        data = data[mask].copy()
        data = data[data[self.seq_col].apply(len) == self.seq_len].reset_index(drop=True)
        return data

    def split_and_save(self, data, output_dir, test_size=0.2, random_state=42):
        y = data[self.label_col].values
        stratify_vals = self.binner.fit_transform(y.reshape(-1, 1))
        
        indices = np.arange(len(data))  # 保留索引对应
        
        # 先划分data和标签以及索引，保持对应
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=stratify_vals
        )
        
        train_df = data.iloc[train_idx].reset_index(drop=True)
        test_df = data.iloc[test_idx].reset_index(drop=True)

        os.makedirs(output_dir, exist_ok=True)
        print("Saving test set...")
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        print("train:",train_df.shape)
        print("test:",test_df.shape)

        return train_df, test_df



    def transform_labels(self, y, fit=False):
        if fit:
            return self.scaler.fit_transform(y.reshape(-1, 1))
        else:
            return self.scaler.transform(y.reshape(-1, 1))

def create_dataset(sequences, y, batch_size, seq_len, ppm_len):
    def generator():
        for i in range(0, len(sequences), batch_size):
            batch_seq = util.one_hot_encode(sequences.iloc[i:i+batch_size])
            batch_ppm = GA_util.prototype_ppms_fast(batch_seq)
            batch_y = y[i:i+batch_size]
            yield (batch_seq, batch_ppm), batch_y

    output_signature = (
        (
            tf.TensorSpec(shape=(batch_size, 4, seq_len), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, ppm_len, ppm_len), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

@hydra.main(config_path="./config", config_name="train")
def main(cfg: DictConfig):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    print("Training parameters:")
    print(f"  batch size: {cfg.train.batch_size}")
    print(f"  epochs: {cfg.train.epoch_num}")
    print(f"  learning rate: {cfg.train.learning_rate}")
    print(f"  latent_dim: {cfg.train.latent_dim}")
    print(f"  input csv: {cfg.data.input_csv}")
    print(f"  split directory: {cfg.data.split_dir}")
    print(f"  model save directory: {cfg.train.save_dir}")

    processor = DataProcessor()
    filtered_data = processor.filter_data(cfg.data.input_csv)
    print("Loading dataset...",filtered_data.shape)
    train_df, test_df = processor.split_and_save(filtered_data, cfg.data.split_dir)
    print("Split train and test")

    y_train = processor.transform_labels(train_df['half_life'].values, fit=True)
    y_test = processor.transform_labels(test_df['half_life'].values)
    
    print("calculating test set ppm")
    
    utr_test = util.one_hot_encode(test_df[['sequence']])
    ppm_test = GA_util.prototype_ppms_fast(utr_test) #error
    
    print("Create training set")
    train_dataset = create_dataset(train_df[['sequence']], y_train, cfg.train.batch_size, processor.seq_len, processor.seq_len)


    resume_path = getattr(cfg.train, 'resume_path', None)
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        joint_model = load_model(resume_path)
    else:
        joint_model = GA_util.create_SANDSTORM(
            seq_len=processor.seq_len,
            ppm_len=processor.seq_len,
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
