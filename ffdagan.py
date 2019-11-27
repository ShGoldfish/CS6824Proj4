'''FeedForward DAGAN on bioinformatics data using Keras
Author: Frank Wanye, Shakiba Davari
Credit: Rowel Atienza
Project: https://github.com/ShGoldfish/CS6824Proj4
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 ffdagan.py
'''

import math
import time
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pickle

# Tell tensorflow warnings to take a hike
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from tensorflow.examples.tutorials.mnist import input_data

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, concatenate
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

class DAGAN(object):
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

        dropout = 0.4
        dim = int(self.img_rows / 16)

        imageIn = Input(shape=(self.img_rows, self.channel,), name="imageIn")
        imageReshape = Reshape((-1,))(imageIn)
        imageDense1 = Dense(dim * 8, activation='sigmoid')(imageReshape)
        imageBatchNorm1 = BatchNormalization(momentum=0.9)(imageDense1)
        imageDropout1 = Dropout(dropout)(imageBatchNorm1)

        imageDense2 = Dense(dim * 4, activation='sigmoid')(imageDropout1)
        imageBatchNorm2 = BatchNormalization(momentum=0.9)(imageDense2)
        imageDropout2 = Dropout(dropout)(imageBatchNorm2)

        imageDense3 = Dense(dim * 2, activation='sigmoid')(imageDropout2)
        imageBatchNorm3 = BatchNormalization(momentum=0.9)(imageDense3)
        imageDropout3 = Dropout(dropout)(imageBatchNorm3)

        imageDense4 = Dense(dim, activation='sigmoid')(imageDropout3)
        imageBatchNorm4 = BatchNormalization(momentum=0.9)(imageDense4)
        imageDropout4 = Dropout(dropout)(imageBatchNorm4)

        imageRepresentation = Dense(100, activation='sigmoid')(imageDropout4)

        noiseIn = Input(shape=(100,), name="noiseIn")
        generatorIn = concatenate([noiseIn, imageRepresentation])
        # dim = 7
        # In: 100
        # Out: dim x dim x depth
        dense1 = Dense(dim, activation='sigmoid')(generatorIn)
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
        generated_image = Reshape((-1, 1), name="output")(dropout5)
        self.G = Model(inputs=[noiseIn, imageIn], outputs=generated_image)
        self.G.summary()

        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        generator = self.generator()
        out_generator = generator.get_layer("output").output
        discriminator = self.discriminator()
        out_discriminator = discriminator(out_generator)
        imageIn = generator.get_layer("imageIn").input
        noiseIn = generator.get_layer("noiseIn").input
        self.AM = Model(inputs=[noiseIn, imageIn], outputs=out_discriminator)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM

class FFDAGAN(object):
    def __init__(self, dataset):
        self.channel = 1
        self.dataset = dataset
        self.data = scipy.io.loadmat("data/{}.mat".format(dataset))

        # TODO: figure out how to split data into train, valid and test sets
        self.x_train = self.data['X']
        img_size = math.ceil(self.x_train.shape[1] / 4.0)
        self.img_rows = int(img_size * 4.0)
        self.img_cols = 1
        if self.img_rows != self.x_train.shape[1]:
            self.x_train = np.pad(self.x_train, ((0,0), (0, self.img_rows - self.x_train.shape[1])), 'minimum')
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, 1).astype(np.float32)

        self.DAGAN = DAGAN(self.img_rows, self.img_cols)
        self.discriminator =  self.DAGAN.discriminator_model()
        self.adversarial = self.DAGAN.adversarial_model()
        self.generator = self.DAGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            # only trains the discriminator: to tell fake images from real ones
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict([noise, images_train])
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            # should only train the generator to learn the patterns for creating a new image
            # in practice, will also train the discriminator
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch([noise, images_train], y)
            log_mesg = "%d: [Discriminator loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [Generator loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = "{}_{}.png".format(self.__class__.__name__, self.dataset)
        image_indices = np.random.randint(0, self.x_train.shape[0], samples)
        images = self.x_train[image_indices, :, :]
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "{}_{}_{}.png".format(self.__class__.__name__, self.dataset, step)
        fake_images = self.generator.predict([noise, images])

        plt.figure(figsize=(5, 4))
        # TODO: increase the width of each "image"
        for image_index in range(images.shape[0]):
            plt.subplot(samples, 2, (image_index*2)+1)
            image = images[image_index, :, :]
            image = np.reshape(image, [self.img_cols, self.img_rows])
            plt.imshow(image, cmap='gray', aspect='auto')
            plt.axis('off')
            plt.subplot(samples, 2, (image_index*2)+2)
            image = fake_images[image_index, :, :]
            image = np.reshape(image, [self.img_cols, self.img_rows])
            plt.imshow(image, cmap='gray', aspect='auto')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def augment(self):
        """Uses the trained dagan to augment the dataset and saves the result to file.
        """
        unique_labels = np.unique(self.data["Y"])
        label_count = dict()
        for unique_label in unique_labels:
            label_count[unique_label] = 0
            for label in self.data["Y"]:
                if label[0] == unique_label:
                    label_count[unique_label] += 1
        target_num_examples = max(label_count.values()) * 2
        new_data = copy(self.data)
        new_data["Real"] = np.asarray([1] * len(new_data["X"]))
        for unique_label in unique_labels:
            num_augmented = target_num_examples - label_count[unique_label]
            noise = np.random.uniform(-1.0, 1.0, size=[num_augmented, 100])
            matching_indices = np.where(self.data["Y"].ravel() == unique_label)
            examples = np.random.choice(matching_indices[0], num_augmented, replace=True)
            images = self.x_train[examples]
            fake_images = self.generator.predict([noise, images])
            fake_labels = np.asarray([unique_label] * num_augmented)
            # [X X X X X X]
            # [[X] [X] [X] [X] [X] [X] [P] [P] [P]]
            # [X X X X X X]
            fake_images_reshaped = fake_images[:, 0:new_data["X"].shape[1], :]
            fake_images_reshaped = fake_images_reshaped.reshape(fake_images.shape[0], new_data["X"].shape[1])
            new_data["X"] = np.concatenate((new_data["X"], fake_images_reshaped))
            new_data["Y"] = np.concatenate((new_data["Y"], np.reshape(fake_labels, (-1, 1))))
            new_data["Real"] = np.concatenate((new_data["Real"], np.asarray([0] * num_augmented)))
        with open("{}_new.pkl".format(self.dataset), 'wb') as dataset:
            pickle.dump(new_data, dataset, protocol=pickle.HIGHEST_PROTOCOL)
    # End of augment()

if __name__ == '__main__':
    for dataset in ["ALLAML", "CLL_SUB_111", "colon", "GLI_85", "GLIOMA", "leukemia", "lung_discrete", "lung", 
                    "lymphoma", "nci9", "Prostate_GE", "SMK_CAN_187", "TOX_171"]:
        print("=====Dataset = {}=====".format(dataset))
        ffdagan = FFDAGAN(dataset)
        timer = ElapsedTimer()
        ffdagan.train(train_steps=50, batch_size=5, save_interval=10)
        timer.elapsed_time()
        ffdagan.plot_images(fake=True, save2file=True)
        ffdagan.plot_images(fake=False, save2file=True)
        ffdagan.augment()
