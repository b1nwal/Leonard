import tensorflow as tf
import base64
import datetime
import os

import pipeline
from hyperparameters import hyperparameters

dataset = pipeline.coord_dataset.take(hyperparameters["train_dataset_size"])




logstr = "logs/profile"
os.makedirs(logstr,exist_ok=True)

def train(model):
    tf.profiler.experimental.start(logdir=logstr)
    session = base64.b64encode(datetime.datetime.now().ctime().encode('utf-8')).decode('utf-8')
    for f,x in enumerate(dataset,1):
        mloss = model.train_step(x)
        percentage = int((f/hyperparameters["train_dataset_size"])*20)
        print("\r["+"="*percentage + ">" + " "*(20-percentage) + "]","Sample: {f}/{h}, Sampl. Loss: {l:.4f}".format(f=f,l=mloss,h=hyperparameters["train_dataset_size"]),end="")

        if f%1000==0:
            model.save(("leo_v1-2-4/" + session + ".keras"))
    tf.profiler.experimental.stop()