from tensorflow.keras.models import load_model
import sys
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")
import GA_util
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
import pandas as pd
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

# Add custom module path
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")
import util
import GA_util

"""max len problem for test and train"""
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


@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    cfg.data.val_ratio
    #Load model 
    model_path = next(Path(cfg.train.best_model_dir).glob('*.h5'), None)
    if model_path is None:
        raise FileNotFoundError(f'No .h5 file found in {cfg.train.best_model_dir}')
    
    joint_model = load_model(model_path)
    print(joint_model.summary())

    # Set parameters
    batch_size = 10
    trainval_test_ratio = cfg.data.holdout_test_ratio

    split_dir = os.path.join(cfg.data.split_dir, f"trainval{1-trainval_test_ratio}_test{trainval_test_ratio}")
    test_path = os.path.join(split_dir, "test.csv")
    dft = pd.read_csv(test_path)
    y_test = dft['half_life_std'].values


        # One-hot encode sequences
    utr_test = util.one_hot_encode(dft[['sequence']])
    indices = np.arange(utr_test.shape[0])

    test_dataset = create_val_dataset(utr_test , y_test, batch_size=batch_size)

    #Predict
    joint_predictions = joint_model.predict(test_dataset)
    y_pred_np = joint_predictions.squeeze()
    y_true_np = y_test.squeeze()
    print(spearmanr(y_true_np, y_pred_np)[0])
