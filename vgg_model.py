import tensorflow as tf
import scipy.io as sio
import numpy as np

class Model:
    def __init__(self, wieght_path, img_h=224, img_w=224, trainable_fc=False):
        vgg_weights = sio.loadmat(wieght_path)

        # self._loaded_layers[layer_idx][0][0][i]
        #   i = 0 -> layer name
        #   i = 1 -> layer type
        #   i = 2 -> weights: [i][0][0] -> W, [i][0][1] -> b
        #   i = 3 -> dim

        self._loaded_layers = vgg_weights['layers'][0]
        self._num_layers = len(self._loaded_layers)
        self._img_h = img_h
        self._img_w = img_w

        # a dictionary to store layer output tensors
        self.tf_layers = {}
        self.output = None
        self.trainable_fc = trainable_fc

    def _get_weights(self, layer_idx):
        layer = self._loaded_layers[layer_idx]
        layer_type = layer[0][0][1][0]
        layer_name = layer[0][0][0][0]

        assert layer_type == 'conv'

        weights = layer[0][0][2]
        w = tf.constant(weights[0][0])
        bias = weights[0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return layer_name, w, bias

    def build_model(self, output=None, save_tb_graph=False):
        if self.output:
            return self.output

        # Avoid using a for-loop to make the network structure clearer
        # Layers of which the indices are skipped are 'relu' activations
        layer = tf.Variable(tf.zeros([1, self._img_h, self._img_w, 3]), dtype='float32')
        self.tf_layers['input'] = layer
        layer = self._conv2d_layer(layer, 0)
        layer = self._conv2d_layer(layer, 2)
        layer = self._pool(layer, 4, pool_type='avg')
        layer = self._conv2d_layer(layer, 5)
        layer = self._conv2d_layer(layer, 7)
        layer = self._pool(layer, 9, pool_type='avg')
        layer = self._conv2d_layer(layer, 10)
        layer = self._conv2d_layer(layer, 12)
        layer = self._conv2d_layer(layer, 14)
        layer = self._conv2d_layer(layer, 16)
        layer = self._pool(layer, 18, pool_type='avg')
        layer = self._conv2d_layer(layer, 19)
        layer = self._conv2d_layer(layer, 21)
        layer = self._conv2d_layer(layer, 23)
        layer = self._conv2d_layer(layer, 25)
        layer = self._pool(layer, 27, pool_type='avg')
        layer = self._conv2d_layer(layer, 28)
        layer = self._conv2d_layer(layer, 30)
        layer = self._conv2d_layer(layer, 32)
        layer = self._conv2d_layer(layer, 34)
        layer = self._pool(layer, 36, pool_type='avg')

        if output is None:
            layer = self._fully_connected_layer(layer, 37)
            layer = self._fully_connected_layer(layer, 39)
            layer = self._fully_connected_layer(layer, 41, is_output=True)
            self.output = layer
        else:
            self.output = self.tf_layers[output]

        return self.output

    def print_layers(self):
        for k, v in self.tf_layers.items():
            print(k, v)

    def _conv2d_layer(self, prev, layer_idx):
        layer_name, filtr, bias = self._get_weights(layer_idx)

        filtr = tf.Variable(filtr, trainable=False, name=layer_name + '_kernel')
        bias = tf.Variable(bias, trainable=False, name=layer_name + '_bias')

        layer = tf.nn.conv2d(prev, filtr, [1, 1, 1, 1], padding='SAME', name=layer_name)
        layer = tf.nn.bias_add(layer, bias)
        layer = tf.nn.relu(layer)
        self.tf_layers[layer_name] = layer
        return layer

    def _pool(self, prev, layer_idx, pool_type='max'):
        layer_name = self._loaded_layers[layer_idx][0][0][0][0]
        if pool_type == 'max':
            layer = tf.nn.max_pool(prev, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        elif pool_type == 'avg':
            layer = tf.nn.avg_pool(prev, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.tf_layers[layer_name] = layer
        return layer
    

    def _fully_connected_layer(self, prev, layer_idx, is_output=False):
        layer_name, W, bias = self._get_weights(layer_idx)

        W = tf.Variable(W, trainable=self.trainable_fc, name=layer_name + '_weights')
        bias = tf.Variable(bias, trainable=self.trainable_fc, name=layer_name + '_bias')

        prev = tf.reshape(prev, [1, -1])
        W = tf.reshape(W, [-1, W.get_shape().as_list()[-1]])
        layer = tf.matmul(prev, W)
        layer = tf.nn.bias_add(layer, bias)

        if is_output:
            layer = tf.nn.softmax(layer)
            
        else:
            layer = tf.nn.relu(layer)
            
        if self.trainable_fc:
            layer = tf.nn.dropout(layer, 0.5)

        self.tf_layers[layer_name] = layer
        return layer


if __name__ == '__main__':
    model = Model('imagenet-vgg-verydeep-19.mat')
    out = model.build_model()
    model.print_layers()
