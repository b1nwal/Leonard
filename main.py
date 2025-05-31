# This is where most things are happening. The model is defined here, all methods are invoked from here.

import tensorflow as tf

import train
from hyperparameters import hyperparameters

class Leonard(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.D2 = tf.keras.layers.Dense(192)
        self.BN2 = tf.keras.layers.BatchNormalization()
        # activation
        
        self.D3 = tf.keras.layers.Dense(192)
        self.BN3 = tf.keras.layers.BatchNormalization()
        # activation
        
        self.D4 = tf.keras.layers.Dense(192)
        self.BN4 = tf.keras.layers.BatchNormalization()
        # activation
        
        self.corpus_callosum = tf.keras.layers.Dropout(hyperparameters["corpus_callosum"])
        
        self.D5 = tf.keras.layers.Dense(192)
        self.BN5 = tf.keras.layers.BatchNormalization()
        # activation
        
        self.D6 = tf.keras.layers.Dense(192)
        self.BN6 = tf.keras.layers.BatchNormalization()
        # activation
        
        self.D7 = tf.keras.layers.Dense(5,kernel_regularizer="l2",dtype="float32")
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])
        self.kicked_in = False
        # self.collecting = false
    def call(self,x,training=False):
        # x /= 5
        y = tf.keras.activations.swish(self.D2(x))
        y = tf.keras.activations.swish(self.D3(y))
        y = tf.keras.activations.swish(self.D4(y))
        y = self.corpus_callosum(y,training=training)
        y = tf.keras.activations.swish(self.D5(y))
        y = tf.keras.activations.swish(self.D6(y))
        y = self.D7(y)
        return y

    @tf.function
    def grads(self, g):
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        

leo = Leonard()

train.train(leo)
