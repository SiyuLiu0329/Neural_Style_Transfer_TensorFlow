import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

def load_image(path, shape=None, bgr=False, preprocess=False, use_crop_or_pad=False):
    """Loads an image from path and provides options to pre-process and normalise it

    Args:
        path (str): Path of the image
        shape (:obj:`list` of 2 :obj: `int`): List of 2 integers representing height and width,
                                      the image will be rescaled as per `shape` if `shape` is
                                      specified
        bgr (bool): The image will be converted to 'bgr' format if True
        preprocess (bool): Subtract the vgg-means from the channels of the image if True
        use_crop_or_pad: Rescale the image by padding or cropping if True, else `BILINEAR` by default
    
    Returns:
        A numpy array of shape (1, height, width, 3)

    """
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
        if use_crop_or_pad:
            image = tf.image.resize_image_with_crop_or_pad(image, shape[0], shape[1])
        else:
            image = tf.image.resize_images(image, [shape[0], shape[1]])

        image = tf.reshape(image, [1, shape[0], shape[1], 3])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            image = image.eval()
            sess.close()

    return image


def predict(prob, path):
    synset = [l.strip() for l in open(path).readlines()]
    pred = np.argsort(prob)[::-1]
    return synset[pred[0]], prob[pred[0]]
