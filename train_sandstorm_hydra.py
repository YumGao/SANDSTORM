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


from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os




class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

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
    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)

    # Load and filter input data
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


class DataProcessor:
    def __init__(self, seq_col='sequence', label_col='half_life', seq_len=130):
        self.seq_col = seq_col
        self.label_col = label_col
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.binner = KBinsDiscretizer(n_bins=10, encode='ordinal')
    
    def filter_data(self, data: pd.DataFrame):
        mask = ~data[self.seq_col].str.upper().str.contains('N', na=False)
        filtered = data[mask].copy()
        filtered = filtered[filtered[self.seq_col].apply(len) == self.seq_len].reset_index(drop=True)
        return filtered
    
    def split_data(self, data: pd.DataFrame, test_size=0.2, random_state=42, stratify=True):
        y = data[self.label_col].values
        indices = np.arange(len(data))
        stratify_val = y if stratify else None
        
        train_df, test_df, y_train, y_test, train_idx, test_idx = train_test_split(
            data, y, indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_val
        )
        return train_df, test_df, y_train, y_test, train_idx, test_idx
    
    def fit_transform_labels(self, y_train):
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1,1))
        y_train_binned = self.binner.fit_transform(y_train.reshape(-1,1))
        return y_train_scaled, y_train_binned
    
    def transform_labels(self, y_test):
        y_test_scaled = self.scaler.transform(y_test.reshape(-1,1))
        y_test_binned = self.binner.transform(y_test.reshape(-1,1))
        return y_test_scaled, y_test_binned
    
    def save_csv(self, df: pd.DataFrame, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

processor = DataProcessor()

# load raw data
data = pd.read_csv(cfg.data.input_csv)

# filter
filtered = processor.filter_data(data)

# split
train_df, test_df, y_train, y_test, train_idx, test_idx = processor.split_data(filtered, stratify=True)

# fit & transform labels on train, transform on test
y_train_scaled, y_train_binned = processor.fit_transform_labels(y_train)
y_test_scaled, y_test_binned = processor.transform_labels(y_test)

# save splits
processor.save_csv(train_df, cfg.data.train_csv)
processor.save_csv(test_df, cfg.data.test_csv)

# one-hot encode train/test sequences for model input
utr_train = util.one_hot_encode(train_df[[processor.seq_col]])
utr_test = util.one_hot_encode(test_df[[processor.seq_col]])
