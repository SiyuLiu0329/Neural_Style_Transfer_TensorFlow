import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np

import tensorflow as tf

def load_image(path, shape=None, show_img=False, bgr=False, preprocess=False):
    image = scipy.misc.imread(path)

    if preprocess:
        if bgr:
            MEAN = [103.939, 116.779, 123.68]
        else:
            MEAN = [123.68, 116.779, 103.939]

        image = image - MEAN

    if bgr:
        image = image[:,:,::-1]
    
    if shape:
        image = tf.image.resize_image_with_crop_or_pad(image, shape[0], shape[1])
        image = tf.reshape(image, [1, shape[0], shape[1], 3])
    else:
        image = tf.convert_to_tensor(image)
        
    if show_img:
        imshow(image)
        plt.show()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image = image.eval()
        sess.close()

    return image


def predict(prob, path):
    synset = [l.strip() for l in open(path).readlines()]
    pred = np.argsort(prob)[::-1]
    return synset[pred[0]], prob[pred[0]]
