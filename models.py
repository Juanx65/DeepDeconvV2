import numpy as np
from functools import partial

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPool2D, UpSampling2D
from tensorflow.keras.layers import Input, Activation, concatenate
from tensorflow.keras.layers import GaussianNoise, GaussianDropout
#from tensorflow.keras.optimizers import Adam
import random as python_random
import matplotlib.pyplot as plt
import time
""" Setting random seeds """
seed = 42

# TensorFlow
tf.random.set_seed(seed)

# Python
python_random.seed(seed)

# NumPy (random number generator used for sampling operations)
rng = np.random.default_rng(seed)




def calc_padding(k, d):
    return d * (k - 1) // 2

class DataGenerator(keras.utils.Sequence):

    def __init__(self, X, win, Nsamples, batch_size=16):

        self.X = X
        self.Nch, self.Nt = X.shape
        self.win = win
        self.Nsamples = Nsamples
        self.batch_size = batch_size

        self.on_epoch_end()

    def __len__(self):
        """ Number of mini-batches per epoch """
        return self.Nsamples // self.batch_size

    def on_epoch_end(self):
        """ Modify data """
        self.__data_generation()
        pass

    def __getitem__(self, idx):
        """ Select a mini-batch """
        batch_size = self.batch_size
        selection = slice(idx * batch_size, (idx + 1) * batch_size)
        samples = self.samples[selection]
        return samples

    def __data_generation(self):
        """ Generate a total batch """

        win = self.win
        X = self.X

        # Number of mini-batches
        N_batch = self.__len__()
        N_total = N_batch * self.batch_size
        # Buffer for mini-batches
        samples = np.zeros((N_total, self.Nch, win, 1))

        inds = rng.integers(low=0, high=self.Nt - win, size=N_total)
        flip_t = rng.integers(low=0, high=2, size=N_total) * 2 - 1
        flip_ch = rng.integers(low=0, high=2, size=N_total) * 2 - 1

        for i, ind in enumerate(inds):
            t_slice = slice(ind, ind + win)
            samples[i, :, :, 0] = X[:, t_slice][::flip_ch[i]][:, ::flip_t[i]]

        self.samples = samples
        pass

class CallBacks:

    @staticmethod
    def tensorboard(logdir):
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir,
            profile_batch=0,
            update_freq="epoch",
            histogram_freq=0,
        )
        return tensorboard_callback

    @staticmethod
    def checkpoint(savefile):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            savefile,
            verbose=1, #antes en cero
            save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            mode="auto",
            update_freq="epoch",
        )
        return checkpoint_callback


################################
# impulse_response  : kernel, in this case chirp
# lam               : hyperparameter (in this case 10)
# f0                : no idea (tiene que ver con la arquitectura del encoder)
# data_shape        : shape de la entrada
# blocks            : layers in encoder decoder
# AA                : Antialiasing ()
# bn                : batch normalization

class UNet(keras.Model):

    def __init__(self, impulse_response, lam=0.0, f0=4, data_shape=(24, 1024, 1), blocks=4, AA=True, bn=False, dropout=0.0):
        super(UNet, self).__init__()

        self.kernel = (3, 5)
        self.f0 = f0
        self.N_blocks = blocks
        self.use_bn = bn

        self.use_dropout = False
        if dropout > 0:
            self.use_dropout = True
            self.dropout_rate = dropout

        self.AA = AA
        self.LR = 5e-4
        self.initializer = keras.initializers.Orthogonal()
        self.activation = tf.keras.activations.swish
        self.data_shape = data_shape

        self.lam = lam
        self.impulse_response = tf.reshape(impulse_response, (1, -1, 1, 1))
        pass

    def call(self, x):
        x_hat = self.AE(x)
        y_hat = tf.nn.conv2d(x_hat, self.impulse_response, padding="SAME", strides=1)#{value for value in variable}
        return x_hat, y_hat

    def compile(self):#, opt):
        super(UNet, self).compile()
        self.opt = tf.keras.optimizers.Adam()
        pass

    def compute_loss(self, Y, Y_hat, X):
        l1 = tf.reduce_mean(tf.abs(X))
        l2 = tf.reduce_mean(tf.square(Y - Y_hat))
        total_loss = l2 + self.lam * l1
        return total_loss, l2, l1

    def train_step(self, Y): #former Y

        if isinstance(Y, tuple):
            Y = Y[0]

        AE = self.AE
        AE_vars = AE.trainable_variables

        with tf.GradientTape() as tape:
            X, Y_hat = self.call(Y)
            total_loss, l2_loss, l1_loss = self.compute_loss(Y, Y_hat, X)

        grads = tape.gradient(total_loss,AE_vars)
        processed_grads = [g for g in grads]
        self.opt.apply_gradients(zip(processed_grads, AE_vars))
        self.compiled_metrics.update_state(Y, Y_hat)
        return {
            "loss": total_loss,
            "l2": l2_loss,
            "l1": l1_loss,
        }

    def test_step(self, Y):
        if isinstance(Y, tuple):
            Y = Y[0]

        X, Y_hat = self.call(Y)
        total_loss, l2_loss, l1_loss = self.compute_loss(Y, Y_hat, X)
        self.compiled_metrics.update_state(Y, Y_hat)
        return {
            "loss": total_loss,
            "l2": l2_loss,
            "l1": l1_loss,
        }

    def conv_layer(self, x, filters, kernel_size, initializer=None,use_bn=False, use_dropout=False, use_bias=True, activ=None):
        """
        Convolution layer > batch normalization > activation > dropout
        """

        if initializer is None:
            initializer = self.initializer

        if use_bn:
            use_bias = False

        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding="same",
            activation=None, kernel_initializer=initializer,
            use_bias=use_bias
        )(x)

        if use_bn:
            x = BatchNormalization()(x)

        if activ is not None:
            x = Activation(activ)(x)

        if use_dropout:
#             x = GaussianDropout(self.dropout_rate)(x)
            x = GaussianNoise(self.dropout_rate)(x)

        return x

    def MaxBlurPool(self, x, kernel_size=(1, 4)):

        if kernel_size[1] == 1:
            a = np.array([1.,])
        elif kernel_size[1] == 2:
            a = np.array([1., 1.])
        elif kernel_size[1] == 3:
            a = np.array([1., 2., 1.])
        elif kernel_size[1] == 4:
            a = np.array([1., 3., 3., 1.])
        elif kernel_size[1] == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif kernel_size[1] == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif kernel_size[1] == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        a = a / a.sum()
        a = np.repeat(a, x.shape[-1]*x.shape[-1])
        a = a.reshape((kernel_size[0], kernel_size[1], x.shape[-1], x.shape[-1]))

        x = MaxPool2D(pool_size=kernel_size, strides=(1, 1))(x)
        x = tf.nn.conv2d(input=x, filters=a, strides=kernel_size, padding="SAME")

        return x

    def construct(self):
        """
        Construct UNet model
        """

        f = self.f0
        kernel = self.kernel
        use_bn = self.use_bn
        use_dropout = self.use_dropout
        AA = self.AA
        activation = self.activation
        data_shape = self.data_shape

        downsamp_size = (2, 4)

        K = 3

        inputs = Input(data_shape)
        x = inputs

        conv_wrap = partial(
            self.conv_layer,
            kernel_size=kernel, use_bn=use_bn,
            use_dropout=use_dropout, activ=activation
        )

        """ Encoder """
        for k in range(K):
            x = conv_wrap(x=x, filters=f)

        x_prev = [x]

        for i in range(self.N_blocks):

            if AA:
                x = self.MaxBlurPool(x, kernel_size=downsamp_size)
            else:
                x = MaxPool2D(pool_size=downsamp_size)(x)

            f = f * 2

            for k in range(K):
                x = conv_wrap(x=x, filters=f)

            x_prev.append(x)

        """ Decoder """
        for i in range(self.N_blocks-1):
            x = UpSampling2D(size=downsamp_size, interpolation="bilinear")(x)
            f = f // 2
            x = concatenate([x, x_prev[-(i+2)]])
            for k in range(K):
                x = conv_wrap(x=x, filters=f)

        x = UpSampling2D(size=downsamp_size, interpolation="bilinear")(x)
        f = f // 2
        x = concatenate([x, x_prev[0]])

        for k in range(K):
                x = conv_wrap(x=x, filters=f)

        x = self.conv_layer(x, filters=1, kernel_size=kernel,
                            use_bn=False, use_dropout=False, use_bias=True, activ="relu")

        self.AE = Model(inputs, x)

        return self.AE
