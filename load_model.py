# This is where most things are happening. The model is defined here, all methods are invoked from here.

import tensorflow as tf
import base64
import datetime
import numpy as np

import pipeline
from hyperparameters import hyperparameters

def eudist(x,y):
    return tf.norm(y-x,axis=1)

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

def build_rotary_matrix(angle):
  r1 = tf.stack([tf.cos(angle), 0., tf.sin(angle)])
  r2 = tf.constant([0.,1.,0.],dtype="float32")
  r3 = tf.stack([-1*tf.sin(angle),0.,tf.cos(angle)])
  return tf.stack([r1,r2,r3])

def build_joint_matrix(angle):
  r1 = tf.constant([1., 0., 0.],dtype="float32")
  r2 = tf.stack([0.,tf.cos(angle),-1*tf.sin(angle)])
  r3 = tf.stack([0.,tf.sin(angle),tf.cos(angle)])
  return tf.stack([r1,r2,r3])

def fwd(rotation_angles): # I am coming back for you: you will be so optimized later dude
    t = tf.transpose(rotation_angles)
    rotary1_matrices = tf.vectorized_map(build_rotary_matrix, t[0],fallback_to_while_loop=False)
    joint1_matrices = tf.matmul(rotary1_matrices, tf.vectorized_map(build_joint_matrix, t[1],fallback_to_while_loop=False))
    joint2_matrices = tf.matmul(joint1_matrices, tf.vectorized_map(build_joint_matrix, t[2],fallback_to_while_loop=False))
    rotary2_matrices = tf.matmul(joint2_matrices, tf.vectorized_map(build_rotary_matrix, t[3],fallback_to_while_loop=False))
    joint3_matrices = tf.matmul(rotary2_matrices, tf.vectorized_map(build_joint_matrix, t[4],fallback_to_while_loop=False))
    
    
    segd = tf.constant([[0,0,0,0,0],[1.17481,1.5,1.,1.,.8],[0,0,0,0,0]],dtype="float32") # TODO change this to match arm measurements
    segd1 = tf.constant([[0],[1.17481],[0]],dtype="float32")
    segd2 = tf.constant([[0],[1.5],[0]],dtype="float32")
    segd3 = tf.constant([[0],[1.],[0]],dtype="float32")
    segd4 = tf.constant([[0],[1.],[0]],dtype="float32")
    segd5 = tf.constant([[0],[.8],[0]],dtype="float32")

    # 117.481
    # 150
    # 200
    seg1 = tf.matmul(rotary1_matrices, segd1)
    seg2 = tf.matmul(joint1_matrices, segd2)
    seg3 = tf.matmul(joint2_matrices, segd3)
    seg4 = tf.matmul(rotary2_matrices, segd4)
    seg5 = tf.matmul(joint3_matrices, segd5)
    
    a = tf.concat([seg1,seg2,seg3,seg4,seg5],axis=2)
    b = tf.reduce_sum(a,axis=2)
    return b

leo = Leonard()
inp = tf.constant([[2,0,0,0,0,0,0,0]],dtype="float32")
leo(inp)



leo.load_weights("leo_v1-2-4\VHVlIEp1biAxMCAwNTo1MDowMSAyMDI1.h5")

a = leo(inp)

print(eudist(fwd(a), inp[:,:3]))