import tensorflow as tf
import numpy as np
from hyperparameters import hyperparameters

# replay = deque(maxlen=hyperparameters["replay_buffer_length"])

batch_size = hyperparameters["batch_size"]
replay_max_prob = hyperparameters["replay_max_probability"]
replay_max_len = hyperparameters["replay_buffer_length"]
train_dataset_size = hyperparameters["train_dataset_size"]

def coord_datagen(): # outputs a batch of coordinates that are approximately uniformly distributed in end effector space.
    for i in range(batch_size * train_dataset_size):  
        # replay_buff_len = len(replay)
        # if (np.random.uniform() < (replay_buff_len / (replay_max_len * replay_max_prob**-1))):
        #     r = replay.pop()
        # else:
        #     r = tf.random.uniform(minval=-np.pi, maxval=np.pi, shape=(5,))
        r = tf.random.uniform(minval=-np.pi, maxval=np.pi, shape=(5,))
        yield r
    
coord_dataset = tf.data.Dataset.from_generator(
    coord_datagen, 
    output_signature=tf.TensorSpec((5,),dtype="float32")
).batch(hyperparameters["batch_size"]).prefetch(tf.data.AUTOTUNE)