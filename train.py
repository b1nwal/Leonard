import tensorflow as tf
import base64
import datetime


import pipeline
from hyperparameters import hyperparameters

dataset = pipeline.coord_dataset.take(hyperparameters["train_dataset_size"])


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

def train(model):
    session = base64.b64encode(datetime.datetime.now().ctime().encode('utf-8'))
    h = hyperparameters["train_dataset_size"]
    for f,x in enumerate(dataset,1):
        fx = fwd(x)
        with tf.GradientTape() as tape:
            y = model.call(fx,training=True)
            loss = eudist(fwd(y),fx)
        gradient = tape.gradient(loss, model.trainable_variables)
        model.grads(gradient)
        # mmax = tf.reduce_max(loss)
        percentage = int((f/hyperparameters["train_dataset_size"])*20)
        mloss = tf.reduce_mean(loss)
        
        print("\r["+"="*percentage + ">" + " "*(20-percentage) + "]","Sample: {f}/{h}, Sampl. Loss: {l:.4f}".format(f=f,l=mloss,h=hyperparameters["train_dataset_size"]),end="")

        if f%1000==0:
            model.save(("leo_v1-2-4/saves/" + session + "")) # TODO make sure to fill this in before saving