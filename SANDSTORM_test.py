from tensorflow.keras.models import load_model
import sys
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")
import GA_util
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr


def create_dataset(x1, y, batch_size):
    def generator():
        for i in range(0, len(x1), batch_size):
            current_batch_size = min(batch_size, len(x1) - i)
            batch_x1 = x1[i:i+batch_size]
            batch_x2 = GA_util.prototype_ppms_fast(batch_x1)
            batch_y = y[i:i+batch_size]
            yield (batch_x1, batch_x2), batch_y
            
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 4, int(x1.shape[2])), dtype=tf.float32),  #  “None” indicates that the batch size is uncertain..
            tf.TensorSpec(shape=(None,int(x1.shape[2]) , int(x1.shape[2])), dtype=tf.float32)  
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  
    )
    
    return tf.data.Dataset.from_generator(
        generator, 
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

#Load model 
model_path = './saved_model/sandstorm_model.h5'
joint_model = load_model(model_path)
#joint_model.summary()

# Set parameters
batch_size = 10


#Use test.npy
utr_test = np.load('./utr_test.npy', allow_pickle=True)
y_test  = np.load('./y_test.npy', allow_pickle=True)
test_dataset = create_dataset(utr_test , y_test, batch_size=batch_size)

#Predict
joint_predictions = joint_model.predict(test_dataset)
y_pred_np = joint_predictions.squeeze()
y_true_np = y_test.squeeze()
print(spearmanr(y_true_np, y_pred_np)[0])