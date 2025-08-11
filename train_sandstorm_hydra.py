# train_sandstorm_hydra.py
# Refactored training script for SANDSTORM with:
# - GPU detection + memory growth
# - Hydra config management
# - Checkpointing with filenames containing val_loss + timestamp
# - Resume training from a saved checkpoint
# - Keeps compatibility with your util and GA_util

import os
import sys
import time
import gc
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Make sure your project src is on PYTHONPATH (same as your original script)
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd

import util
import GA_util

print("Modules imported.")


def enable_gpu_mem_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print("Could not set memory growth:", e)
        print(f"Using GPU(s): {len(gpus)}")
    else:
        print("No GPU found, using CPU")


def create_dataset(x1, y, batch_size, seq_len, ppm_len):
    # generator yields variable-sized final batch so use None for first dim
    def generator():
        for i in range(0, len(x1), batch_size):
            batch_x1 = x1[i:i+batch_size]
            batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
            batch_y = y[i:i+batch_size]
            yield (batch_x1.astype('float32'), batch_x2.astype('float32')), batch_y.astype('float32')

    output_signature = (
        (
            tf.TensorSpec(shape=(None, 4, seq_len), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ppm_len, ppm_len), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Hydra config:\n", cfg)

    enable_gpu_mem_growth()

    # --- Config parameters ---
    epoch_num = int(cfg.train.epoch_num)
    iterations = int(cfg.train.iterations)
    latent_dim = int(cfg.train.latent_dim)
    learning_rate = float(cfg.train.learning_rate)
    batch_size = int(cfg.train.batch_size)
    resume_ckpt = cfg.train.resume_ckpt if 'resume_ckpt' in cfg.train else None
    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data path
    data_path = cfg.data.input_csv

    # --- Load & preprocess ---
    data = pd.read_csv(data_path)
    mask = ~data['sequence'].str.upper().str.contains('N', na=False)
    data = data[mask].copy()
    data = data[data['sequence'].apply(len) == 130].reset_index(drop=True)

    seq_len = len(data['sequence'].iloc[0])
    ppm_len = seq_len

    y = data['half_life'].values

    # Label transform (keep same behaviour)
    from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
    stand = StandardScaler()
    y_transformed = stand.fit_transform(y.reshape(-1, 1))

    est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    encoded_vals = est.fit_transform(y.reshape(-1, 1))

    utrs = util.one_hot_encode(data[['sequence']])
    indices = np.arange(utrs.shape[0])

    # Fixed split with stratify
    from sklearn.model_selection import train_test_split
    utr_train, utr_test, y_train, y_test, _, _ = train_test_split(
        utrs, y_transformed, indices, test_size=0.2, stratify=encoded_vals, random_state=42)

    # Save test set for evaluation / later use
    np.save('utr_test.npy', utr_test)
    np.save('y_test.npy', y_test)

    # Ensure training set divisible by batch_size (optional)
    num_samples = (len(utr_train) // batch_size) * batch_size
    utr_train, y_train = utr_train[:num_samples], y_train[:num_samples]

    train_dataset = create_dataset(utr_train, y_train, batch_size, seq_len, ppm_len)

    # Precompute ppm for validation (full array)
    ppm_test = GA_util.prototype_ppms_fast(utr_test).astype('float32')

    # --- Training loop(s) ---
    for it in range(iterations):
        print(f"Starting iteration {it+1}/{iterations}")

        # Create model via your util (assumed returns a compiled or uncompiled model)
        model = GA_util.create_SANDSTORM(
            seq_len=seq_len,
            ppm_len=ppm_len,
            latent_dim=latent_dim,
            internal_activation=cfg.model.internal_activation
        )

        # If resume checkpoint specified AND file exists -> load full model
        if resume_ckpt:
            resume_path = Path(resume_ckpt)
            if resume_path.exists():
                print(f"Resuming from checkpoint: {resume_path}")
                # load_model restores optimizer state only if model was saved with optimizer
                model = tf.keras.models.load_model(str(resume_path))
            else:
                print(f"Resume checkpoint not found: {resume_path}. Continuing from scratch.")

        # Compile (if load_model returned a compiled model, this will recompile)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        # Callbacks: memory cleanup, early stopping, ModelCheckpoint
        class MemoryCleanupCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # ModelCheckpoint with formatted filename that includes val_loss and epoch
        ckpt_pattern = str(save_dir / f"sanstorm_epoch{{epoch:02d}}-val{{val_loss:.4f}}-{timestamp}.h5")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_pattern,
            monitor='val_loss',
            save_best_only=cfg.train.save_best_only,
            save_weights_only=False,  # save full model so we can resume
            mode='min',
            verbose=1
        )

        callbacks = [MemoryCleanupCallback(), checkpoint_cb]

        if cfg.train.early_stopping.enabled:
            es_cfg = cfg.train.early_stopping
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(es_cfg.patience), restore_best_weights=bool(es_cfg.restore_best_weights)))

        # Fit
        history = model.fit(
            train_dataset,
            epochs=epoch_num,
            validation_data=([utr_test.astype('float32'), ppm_test], y_test.astype('float32')),
            callbacks=callbacks
        )

        # After training, save the final model with last val_loss in filename
        final_val_loss = history.history.get('val_loss', [None])[-1]
        final_ts = time.strftime("%Y%m%d-%H%M%S")
        final_name = save_dir / f"sanstorm_final_val{final_val_loss:.4f}_{final_ts}.h5"
        model.save(str(final_name))
        print(f"Saved final model to {final_name}")


if __name__ == '__main__':
    main()
