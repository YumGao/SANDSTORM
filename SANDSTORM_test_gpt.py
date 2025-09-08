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
import joblib
import tensorflow_addons as tfa

# Add custom module path
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")
import util
import GA_util


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

    def __init__(self, cfg, scaler):
        self.cfg = cfg
        self.scaler = scaler
        self.max_len = None
        self.test_df = None
        self.seq_test = None
        self.y_test_orig = None  # ✅ only keep original labels

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
        """Encode sequences; keep labels in original scale."""
        self.seq_test = util.one_hot_encode(self.test_df[['sequence']])
        self.y_test_orig = self.test_df['half_life'].values  # ✅ original values
        return self.seq_test, self.y_test_orig

    def create_test_dataset(self, batch_size):
        """Create tf.data.Dataset for test (labels not needed here)."""
        def generator():
            for i in range(0, len(self.seq_test), batch_size):
                batch_x1 = self.seq_test[i:i+batch_size]
                batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
                # dummy labels just to fit dataset structure
                yield (batch_x1, batch_x2), np.zeros((len(batch_x1), 1), dtype=np.float32)

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

    def inverse_transform(self, y_scaled):
        """Inverse transform scaled predictions back to original scale."""
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


@hydra.main(config_path="./config", config_name="test", version_base=None)
def main(cfg: DictConfig):
    setup_gpu()
    print("Hydra configuration loaded:")
    print(cfg)

    # Load scaler
    scaler_files = glob.glob(os.path.join(cfg.test.best_model_path, "sandstorm_scaler.pkl"))
    if not scaler_files:
        raise FileNotFoundError(f"No 'sandstorm_scaler.pkl' found in {cfg.test.best_model_path}")
    scaler_path = scaler_files[0]
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")

    # Load model with AdamW
    model_files = glob.glob(os.path.join(cfg.test.best_model_path, "*.h5"))
    if not model_files:
        raise FileNotFoundError(f"No .h5 model files found in {cfg.test.best_model_path}")
    if len(model_files) > 1:
        print(f"Warning: Multiple .h5 files found. Using the first: {model_files[0]}")
    model_path = model_files[0]
    with tf.keras.utils.custom_object_scope({'AdamW': tfa.optimizers.AdamW}):
        joint_model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Prepare test data
    test_manager = TestDatasetManager(cfg, scaler)
    test_df, max_len = test_manager.load_and_filter_test_data()
    seq_test, y_test_orig = test_manager.prepare_test_features()

    # Create test dataset
    test_dataset = test_manager.create_test_dataset(cfg.data.batch_size)

    # Predict (scaled) then inverse transform
    y_pred_scaled = joint_model.predict(test_dataset, verbose=1)
    y_pred = test_manager.inverse_transform(y_pred_scaled)

    # Metrics on original scale
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    spearman_r, _ = spearmanr(y_test_orig, y_pred)

    print(f"Test Metrics:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  Spearman R: {spearman_r:.3f}")

    # Save metrics
    os.makedirs(cfg.test.output_dir, exist_ok=True)
    metrics_df = pd.DataFrame({
        'mse': [mse],
        'mae': [mae],
        'spearman_r': [spearman_r],
        'model_path': [model_path],
        'scaler_path': [scaler_path]
    })
    metrics_path = os.path.join(cfg.test.output_dir, f"{cfg.model_name}_test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Visualization 1: Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred, alpha=0.5, label='Predictions')
    plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)],
             'r--', label='Ideal')
    plt.xlabel('True Half-Life')
    plt.ylabel('Predicted Half-Life')
    plt.title(f'True vs Predicted Half-Life (Spearman R: {spearman_r:.3f})')
    plt.legend()
    scatter_plot_path = os.path.join(cfg.test.output_dir, f"{cfg.model_name}_true_vs_pred_scatter.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_plot_path}")

    # Visualization 2: Residual plot
    residuals = y_test_orig - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, label='Residuals')
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
