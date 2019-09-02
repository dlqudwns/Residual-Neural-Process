
import tensorflow as tf
import numpy as np
from array import array
import os
import imageio
import scipy

MNIST_TEST_PATH = ''
MNIST_TRAIN_PATH = ''
CELEBA_TRAIN_PATH= ''
CELEBA_TEST_FILE = ''
if not MNIST_TEST_PATH or not MNIST_TRAIN_PATH or not CELEBA_TRAIN_PATH or not CELEBA_TEST_FILE:
    raise NotImplementedError('dataset path not set!')

class ImagesReader(object):

    def __init__(self,
                 batch_size,
                 dataset_name,
                 max_num_context= 200,
                 include_context = True,
                 min_num_context = 3,
                 testing_only = False):

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self.include_context = include_context
        self.min_num_context = min_num_context

        if dataset_name == 'mnist':
            with open(MNIST_TRAIN_PATH, 'rb') as file:
                file.read(16)
                image_data = array('B', file.read())
                training_data = np.reshape(image_data, [60000, 28, 28, 1]).astype(np.float32)
                self.training_data = (training_data / 255.0 - 0.5)
            with open(MNIST_TEST_PATH, 'rb') as file:
                file.read(16)
                image_data = array('B', file.read())
                test_data = np.reshape(image_data, [10000, 28, 28, 1]).astype(np.float32)
                self.test_data = (test_data / 255.0 - 0.5)
            self.size = 28
            self.dy = 1
        elif dataset_name == 'celeba':
            if not testing_only:
                path = CELEBA_TRAIN_PATH
                filenames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                images = []
                for fn in filenames:
                    arr = np.asarray(imageio.imread(fn))
                    resized = scipy.misc.imresize(arr, (32, 32))
                    scaled = (resized / 255.0 - 0.5)
                    images.append(scaled)
                images = np.array(images, dtype=np.float32)
                self.training_data = images[:200000]
                self.test_data = images[200000:200100]
            else:
                filename = CELEBA_TEST_FILE
                arr = np.asarray(imageio.imread(filename))
                resized = scipy.misc.imresize(arr, (32, 32))
                scaled = (resized / 255.0 - 0.5)
                images = np.array([scaled], dtype=np.float32)
                self.training_data=self.test_data=images
            self.size = 32
            self.dy = 3
        else:
            raise NotImplementedError
        self.train_placeholder = tf.placeholder(self.training_data.dtype, self.training_data.shape)
        self.test_placeholder = tf.placeholder(self.test_data.dtype, self.test_data.shape)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_placeholder))\
            .repeat().shuffle(100).batch(self._batch_size).make_initializable_iterator()
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_placeholder))\
            .repeat().make_initializable_iterator()

    def initialize(self, sess):
        sess.run([self.train_dataset.initializer, self.test_dataset.initializer],
                 feed_dict={self.train_placeholder: self.training_data,
                            self.test_placeholder: self.test_data})

    def generate_data(self, testing):
        num_context = tf.random_uniform(
            shape=[], minval=self.min_num_context, maxval=self._max_num_context, dtype=tf.int32)

        if testing:
            num_target = self.size * self.size
            num_total_points = num_target
            ind = np.mgrid[:self.size, :self.size].reshape(2, -1).T # s^2 x 2

            x_values = tf.convert_to_tensor(ind[None] * 2.0 / self.size - 1.0) # 1 x s^2 x 2
            y_values = tf.gather_nd(self.test_dataset.get_next(), ind)[None]
        else:
            minval = 3
            maxval = self._max_num_context - num_context + 3 if self.include_context else self._max_num_context
            num_target = tf.random_uniform(
                shape=(), minval=minval, maxval=maxval, dtype=tf.int32)
            num_total_points = num_context + num_target

            ind = tf.random_uniform([self._batch_size, num_total_points, 2], maxval=32, dtype=tf.int32)
            x_values = tf.cast(ind, tf.float32) * 2.0 / self.size - 1.0
            batch = self.train_dataset.get_next()
            y_values = tf.map_fn(lambda x: tf.gather_nd(x[0], x[1]), (batch, ind), dtype=tf.float32)

        if testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = tf.random_shuffle(tf.range(num_target))
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            if self.include_context:
                target_x = x_values[:, :num_context+num_target, :]
                target_y = y_values[:, :num_context+num_target, :]
            else:
                target_x = x_values[:, num_context:num_context+num_target, :]
                target_y = y_values[:, num_context:num_context+num_target, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        return context_x, context_y, target_x, target_y