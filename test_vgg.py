from utilities import load_image
import tensorflow as tf
from vgg_model import Model
import numpy as np
from utilities import predict

# Load image, need to specify the path
image = load_image('img/gr.jpg', shape=[224, 224], preprocess=True, bgr=False)
model = Model('imagenet-vgg-verydeep-19.mat')
out = model.build_model()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(model.tf_layers['input'].assign(image))
    prob = sess.run(out)
    # Need to read 'synset.txt' file to print out the name of the class
    pred, prob = predict(prob[0], './synset.txt')
    print(pred, prob)
