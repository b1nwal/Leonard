# This is where most things are happening. The model is defined here, all methods are invoked from here.

import tensorflow as tf
import base64
import datetime
import os
import numpy as np

import pipeline
from hyperparameters import hyperparameters

@tf.function
def eudist(x,y):
    return tf.norm(y-x,axis=1)

@tf.function
def build_rotary_matrix(angle):
  r1 = tf.stack([tf.cos(angle), 0., tf.sin(angle)])
  r2 = tf.constant([0.,1.,0.],dtype="float32")
  r3 = tf.stack([-1*tf.sin(angle),0.,tf.cos(angle)])
  return tf.stack([r1,r2,r3])
@tf.function
def build_joint_matrix(angle):
  r1 = tf.constant([1., 0., 0.],dtype="float32")
  r2 = tf.stack([0.,tf.cos(angle),-1*tf.sin(angle)])
  r3 = tf.stack([0.,tf.sin(angle),tf.cos(angle)])
  return tf.stack([r1,r2,r3])
@tf.function
def fwd(rotation_angles): # I am coming back for you: you will be so optimized later dude
    t = tf.transpose(rotation_angles)
    rotary1_matrices = tf.vectorized_map(build_rotary_matrix, t[0],fallback_to_while_loop=False)
    joint1_matrices = tf.matmul(rotary1_matrices, tf.vectorized_map(build_joint_matrix, t[1],fallback_to_while_loop=False))
    joint2_matrices = tf.matmul(joint1_matrices, tf.vectorized_map(build_joint_matrix, t[2],fallback_to_while_loop=False))
    rotary2_matrices = tf.matmul(joint2_matrices, tf.vectorized_map(build_rotary_matrix, t[3],fallback_to_while_loop=False))
    joint3_matrices = tf.matmul(rotary2_matrices, tf.vectorized_map(build_joint_matrix, t[4],fallback_to_while_loop=False))
    
    segd = tf.constant([[0],[1],[0]],dtype="float32") # TODO change this to match arm measurements
    
    seg1 = tf.matmul(rotary1_matrices, segd)
    seg2 = tf.matmul(joint1_matrices, segd)
    seg3 = tf.matmul(joint2_matrices, segd)
    seg4 = tf.matmul(rotary2_matrices, segd)
    seg5 = tf.matmul(joint3_matrices, segd)
    
    a = tf.concat([seg1,seg2,seg3,seg4,seg5],axis=2)
    b = tf.reduce_sum(a,axis=2)
    return b

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
        

    @tf.function
    def train_step(self, f, x, rs):
        with tf.GradientTape() as tape:
            fx = fwd(x)
            y = self.call(tf.concat((fx,rs),axis=1),training=True)
            rotsum = tf.reduce_sum(abs(rs - y),axis=1)
            dist = eudist(fwd(y),fx)
            loss = dist + (hyperparameters["motion_penalty_multiplier"] * (f / hyperparameters["train_dataset_size"])) * rotsum
        gradient = tape.gradient(loss, self.trainable_variables)
        self.grads(gradient)
        # mmax = tf.reduce_max(loss)
        return (tf.reduce_mean(dist), tf.reduce_mean(loss), tf.reduce_mean(rotsum))


leo = Leonard()

dataset = pipeline.coord_dataset.take(hyperparameters["train_dataset_size"])
np.set_printoptions(threshold=1000, linewidth=200)   # set wide line

session = base64.b64encode(datetime.datetime.now().ctime().encode('utf-8')).decode('utf-8')
for f,x in enumerate(dataset,1):
    dist, mloss, rd = leo.train_step(tf.constant([f],dtype="float32"), x[:,:5],x[:,5:])
    percentage = int((f/hyperparameters["train_dataset_size"])*20)
    print("\r["+"="*percentage + ">" + " "*(20-percentage) + "]","Sample: {f}/{h}, Loss: {m:.2f}, Rotation Delta: {rd:.2f}, Dist: {dist:.2f}".format(dist=dist,rd=rd,f=f,m=mloss,h=hyperparameters["train_dataset_size"]),end="")

    if f%1000==0:
        leo.save(("leo_v1-2-4/" + session + ".keras"))