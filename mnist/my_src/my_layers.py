'''
Custom keras layers for doing low-dimensional convolutional filter operations.

https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models
https://www.tensorflow.org/beta/guide/keras/saving_and_serializing
'''

import numpy as np
import tensorflow as tf

class LowDimConv2D(tf.keras.layers.Layer):

    def __init__(self, filter_basis, n_out_channels, activation = None, **kwargs):
        super(LowDimConv2D, self).__init__(**kwargs)

        self.filter_basis = filter_basis
        self.n_out_channels = n_out_channels
        self.activation = activation

        # Will be created during call to build().

        self.filter_space_projection = None
        self.filter_space_weights = None
        self.bias = None

    def build(self, input_shape):
        n_input_channels = input_shape[-1]
        repeated_filter_shape = (self.filter_basis.shape[0],
                                 self.filter_basis.shape[1],
                                 n_input_channels,
                                 self.filter_basis.shape[2])
        repeated_basis = np.broadcast_to(self.filter_basis[..., np.newaxis, :],
                                         repeated_filter_shape)
        filter_init = tf.keras.initializers.Constant(repeated_basis)

        self.filter_space_projection = tf.keras.layers.DepthwiseConv2D(
            kernel_size = self.filter_basis.shape[:2],
            depth_multiplier = self.filter_basis.shape[-1],
            padding = 'same',
            use_bias = False,
            depthwise_initializer = filter_init,
            trainable = False,
            name = 'filter_space_projection')

        self.filter_space_weights = self.add_variable(
            'filter_space_weights',
            shape = [int(input_shape[-1] * self.filter_basis.shape[-1]), self.n_out_channels])
        self.bias = self.add_variable('bias', shape = [self.n_out_channels])

    def call(self, inputs):
        filter_space_coords = self.filter_space_projection(inputs)
        output = tf.tensordot(filter_space_coords, self.filter_space_weights, axes = [[-1], [0]]) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(LowDimConv2D, self).get_config()
        config.update({'filter_basis' : self.filter_basis,
                       'n_out_channels' : self.n_out_channels})
        if self.activation is None:
            config.update({'activation' : self.activation})
        else:
            config.update({'activation' : self.activation._tf_api_names[0]})

        return config

    @classmethod
    def from_config(cls, config):
        output = cls(**config) 
        output.filter_basis = np.array(output.filter_basis)

        # If the activation function is function name string, then use that
        # to get the function from tensorflow.
        if output.activation.__class__ == ''.__class__:
            module_list = output.activation.split('.')
            x = tf
            for obj in module_list:
                x = getattr(x, obj)
            output.activation = x 
        return output

class AnisotropicGrad2D(tf.keras.layers.Layer):
    '''
    Train on direction (not necessarily normalized) direction self.first_direction.
    Other trainable weight is the log of the scaling to apply to product with the
    rotation of self.first_direction.
    ''' 

    def __init__(self, n_out_channels, **kwargs):
        super(AnisotropicGrad2D, self).__init__(**kwargs)
        self.n_out_channels = n_out_channels

    def build(self, input_shape): 
        self.angles = self.add_variable('angles',
                                        shape = [self.n_out_channels],
                                        initializer = tf.keras.initializers.RandomUniform(0, np.pi/2))

        # self.all_log_scale = self.add_variable('all_log_scale',
        #                                        shape = [self.n_out_channels],
        #                                        initializer = tf.keras.initializers.RandomUniform)
        self.other_log_scale = self.add_variable('other_log_scale',
                                                 shape = [self.n_out_channels],
                                                 initializer = tf.keras.initializers.RandomUniform)
        self.rotation = tf.constant([[0, -1],
                                     [1, 0]], dtype = 'float32')

    def call(self, xs):
        # Don't forget that xs.shape[0] is the number of input samples.
        # xs should be a 2D image of channels.
        differences = tf.stack([xs[:, 1:, :-1, ...] - xs[:, :-1, :-1, ...],
                                xs[:, :-1, 1:, ...] - xs[:, :-1, :-1, ...]],
                                axis = -1)
        first_directions = tf.stack([tf.math.cos(self.angles), tf.math.sin(self.angles)],
                                    axis = 0)
        first_square = tf.tensordot(differences, first_directions, axes = [[-1], [0]])
        first_square = tf.math.square(first_square)

        second_directions = tf.stack([-tf.math.sin(self.angles), tf.math.cos(self.angles)], axis = 0)
         #tf.tensordot(self.rotation, first_directions, axes = [[-1], [0]])
        second_square = tf.tensordot(differences, second_directions, axes = [[-1], [0]]) 
        second_square = tf.math.square(second_square) * tf.math.exp(self.other_log_scale)
        sizes = tf.math.sqrt(1 + first_square + second_square) #* tf.math.exp(self.all_log_scale)
        return sizes

