import tensorflow as tf
import numpy as np

from hyperparameters import hyperparameters

# replay = deque(maxlen=hyperparameters["replay_buffer_length"])

batch_size = hyperparameters["batch_size"]
replay_max_prob = hyperparameters["replay_max_probability"]
replay_max_len = hyperparameters["replay_buffer_length"]
train_dataset_size = hyperparameters["train_dataset_size"]

with tf.device("/GPU:0"):
    data_tensor = tf.random.uniform(minval=-np.pi, maxval=np.pi, shape=(hyperparameters["batch_size"] * hyperparameters["train_dataset_size"],5))

    coord_dataset = tf.data.Dataset.from_tensor_slices(data_tensor).batch(hyperparameters["batch_size"]).prefetch(tf.data.AUTOTUNE)