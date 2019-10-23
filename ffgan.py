'''DCGAN on bioinformatics data using Keras
Author: Frank Wanye, Shakiba Davari
Credit: Rowel Atienza
Project: https://github.com/ShGoldfish/CS6824Proj4
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
# from tensorflow.examples.tutorials.mnist import input_data

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.channel)
        self.D.add(Dense(int(self.img_rows / 2), input_shape=input_shape, activation='tanh'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Dropout(dropout))
        self.D.add(Dense(int(self.img_rows / 4), activation='tanh'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Dropout(dropout))
        self.D.add(Dense(int(self.img_rows / 8), activation='tanh'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Dropout(dropout))
        self.D.add(Dense(int(self.img_rows / 16), activation='tanh'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        # self.G = Sequential()
        noiseIn = Input(shape=(100,))
        dropout = 0.4
        depth = 64 + 64 + 64 + 64  # why is this the depth?
        # dim = 7
        dim = int(self.img_rows / 16)
        # In: 100
        # Out: dim x dim x depth
        dense1 = Dense(dim, activation='sigmoid')(noiseIn)
        batchnorm1 = BatchNormalization(momentum=0.9)(dense1)
        dropout1 = Dropout(dropout)(batchnorm1)

        dense2 = Dense(dim * 2, activation='sigmoid')(dropout1)
        batchnorm2 = BatchNormalization(momentum=0.9)(dense2)
        dropout2 = Dropout(dropout)(batchnorm2)

        dense3 = Dense(dim * 4, activation='sigmoid')(dropout2)
        batchnorm3 = BatchNormalization(momentum=0.9)(dense3)
        dropout3 = Dropout(dropout)(batchnorm3)

        dense4 = Dense(dim * 8, activation='sigmoid')(dropout3)
        batchnorm4 = BatchNormalization(momentum=0.9)(dense4)
        dropout4 = Dropout(dropout)(batchnorm4)

        dense5 = Dense(self.img_rows, activation='linear')(dropout4)
        batchnorm5 = BatchNormalization(momentum=0.9)(dense5)
        dropout5 = Dropout(dropout)(batchnorm5)
        generated_image = Reshape((-1, 1))(dropout5)
        self.G = Model(inputs=noiseIn, outputs=generated_image)
        self.G.summary()

        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class FFGAN(object):
    def __init__(self, dataset):
        self.channel = 1
        self.dataset = dataset
        data = scipy.io.loadmat("data/{}.mat".format(dataset))

        # TODO: figure out how to split data into train, valid and test sets
        self.x_train = data['X']
        img_size = math.ceil(self.x_train.shape[1] / 4.0)
        self.img_rows = int(img_size * 4.0)
        self.img_cols = 1
        if self.img_rows != self.x_train.shape[1]:
            self.x_train = np.pad(self.x_train, ((0,0), (0, self.img_rows - self.x_train.shape[1])), 'minimum')
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, 1).astype(np.float32)

        self.DCGAN = DCGAN(self.img_rows, self.img_cols)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            # only trains the discriminator: to tell fake images from real ones
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            # should only train the generator to learn the patterns for creating a new image
            # in practice, will also train the discriminator
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [Discriminator loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [Generator loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = "{}_{}_real.png".format(self.__class__.__name__, self.dataset)
        if fake:
            filename="{}_{}_fake.png".format(self.__class__.__name__, self.dataset)
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "{}_{}_fake_{}.png".format(self.__class__.__name__, self.dataset, step)
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :]

        plt.figure(figsize=(2, 4))
        # TODO: increase the width of each "image"
        for i in range(images.shape[0]):
            plt.subplot(samples, 1, i+1)
            image = images[i, :, :]
            image = np.reshape(image, [self.img_cols, self.img_rows])
            plt.imshow(image, cmap='gray', aspect='auto')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = FFGAN("ALLAML")
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10, batch_size=1, save_interval=10)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True, save2file=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)