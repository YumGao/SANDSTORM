import sys
import os
import glob
import hydra
from omegaconf import DictConfig
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

# Add custom module path
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")
import util
import GA_util
Final Epoch Metrics (Epoch 114):
  Train Loss (MSE): 0.059, Train MAE: 0.161
  Val Loss (MSE): 0.055, Val MAE: 0.157
  Test Loss (MSE): 0.053, Test MAE: 0.159, Test Spearman R: nan
Metrics saved to ./sandstorm_metrics_sprnan.csv
[2025-09-08 13:14:28,983][nupack.rebind.render][INFO] - cleaning up Python C++ resources
Why the spearman R is nan?
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU(s): {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"Error setting up GPU memory growth: {e}")
    else:
        print("No GPU detected, running on CPU.")

class TestDatasetManager:
    """Manages loading and processing of test data, aligned with training."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_len = None
        self.test_df = None
        self.seq_test = None
        self.y_test = None
        self.y_test_orig = None
    
    def load_and_filter_test_data(self):
        """Load and filter the test DataFrame, matching training preprocessing."""
        df = pd.read_csv(self.cfg.test.test_csv)
        mask = ~df['sequence'].str.upper().str.contains('N', na=False)
        df = df[mask].copy()
        df = df[df['sequence'].apply(len) == 201].reset_index(drop=True)
        self.max_len = df["sequence"].str.len().max()
        self.test_df = df
        print(f"Loaded and filtered test data: {self.test_df.shape}")
        return self.test_df, self.max_len
    
    def prepare_test_features(self):
        """Encode sequences and use raw labels."""
        self.seq_test = util.one_hot_encode(self.test_df[['sequence']])
        self.y_test_orig = self.test_df['half_life'].values  # Original for evaluation
        self.y_test = self.test_df['half_life'].values.reshape(-1, 1)  # Raw labels for dataset compatibility
        return self.seq_test, self.y_test_orig
    
    def create_test_dataset(self, batch_size):
        """Create tf.data.Dataset for test, matching training dataset structure."""
        def generator():
            for i in range(0, len(self.seq_test), batch_size):
                batch_x1 = self.seq_test[i:i+batch_size]
                batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
                batch_y = self.y_test[i:i+batch_size]  # Raw y, included for dataset compatibility
                yield (batch_x1, batch_x2), batch_y
        
        output_signature = (
            (
                tf.TensorSpec(shape=(None, 4, self.max_len), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.max_len, self.max_len), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
        return tf.data.Dataset.from_generator(
            generator, 
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

@hydra.main(config_path="./config", config_name="test", version_base=None)
def main(cfg: DictConfig):
    setup_gpu()
    print("Hydra configuration loaded:")
    print(cfg)
    
    # Find and load model from best_model_path with custom AdamW optimizer
    model_files = glob.glob(os.path.join(cfg.test.best_model_path, "*.h5"))
    if not model_files:
        raise FileNotFoundError(f"No .h5 model files found in {cfg.test.best_model_path}")
    if len(model_files) > 1:
        print(f"Warning: Multiple .h5 files found in {cfg.test.best_model_path}. Using the first one: {model_files[0]}")
    model_path = model_files[0]
    with tf.keras.utils.custom_object_scope({'AdamW': tfa.optimizers.AdamW}):
        joint_model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # Initialize test dataset manager
    test_manager = TestDatasetManager(cfg)
    test_df, max_len = test_manager.load_and_filter_test_data()
    seq_test, y_test_orig = test_manager.prepare_test_features()
    
    # Create test dataset
    test_dataset = test_manager.create_test_dataset(cfg.data.batch_size)
    
    # Obtain predictions (raw scale)
    y_pred = joint_model.predict(test_dataset, verbose=1)
    
    # Compute performance metrics on raw scale
    mse = mean_squared_error(y_test_orig, y_pred.flatten())
    mae = mean_absolute_error(y_test_orig, y_pred.flatten())
    spearman_r, _ = spearmanr(y_test_orig, y_pred.flatten())
    
    # Print metrics
    print(f"Test Metrics:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  Spearman R: {spearman_r:.3f}")
    
    # Save metrics to CSV
    os.makedirs(cfg.test.output_dir, exist_ok=True)
    metrics_df = pd.DataFrame({
        'mse': [mse],
        'mae': [mae],
        'spearman_r': [spearman_r],
        'model_path': [model_path]
    })
    metrics_path = os.path.join(cfg.test.output_dir, f"{cfg.model_name}_test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # Visualization 1: Scatter plot of true vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred.flatten(), alpha=0.5, label='Predictions')
    plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], 'r--', label='Ideal')
    plt.xlabel('True Half-Life')
    plt.ylabel('Predicted Half-Life')
    plt.title(f'True vs Predicted Half-Life (Spearman R: {spearman_r:.3f})')
    plt.legend()
    scatter_plot_path = os.path.join(cfg.test.output_dir, f"{cfg.model_name}_true_vs_pred_scatter.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_plot_path}")
    
    # Visualization 2: Residual plot
    residuals = y_test_orig - y_pred.flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred.flatten(), residuals, alpha=0.5, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
    plt.xlabel('Predicted Half-Life')
    plt.ylabel('Residual (True - Predicted)')
    plt.title('Residual Plot')
    plt.legend()
    residual_plot_path = os.path.join(cfg.test.output_dir, f"{cfg.model_name}_residual_plot.png")
    plt.savefig(residual_plot_path)
    plt.close()
    print(f"Residual plot saved to {residual_plot_path}")

if __name__ == "__main__":
    main()