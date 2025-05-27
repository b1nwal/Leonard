import tensorflow as tf

@tf.function
def eudist(x,y):
    return tf.norm(y-x,axis=1)