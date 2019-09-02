
# This file is modified version of part of code from:
# https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb

import tensorflow as tf

class GPCurvesReader(object):
    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 l1_scale=0.6,
                 sigma_scale=1.0,
                 testing=False,
                 include_context=True,
                 random_kernel = False,
                 min_num_context = 3):

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing
        self.include_context = include_context
        self.random_kernel = random_kernel
        self.min_num_context = min_num_context

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        num_total_points = tf.shape(xdata)[1]

        # Expand and take the difference
        xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
        xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        norm = tf.reduce_sum(
            norm, -1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * tf.eye(num_total_points)

        return kernel

    def generate_curves(self):
        num_context = tf.random_uniform(
            shape=[], minval=self.min_num_context, maxval=self._max_num_context, dtype=tf.int32)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = tf.tile(
                tf.expand_dims(tf.range(-20., 20., 1. / 10, dtype=tf.float32), axis=0),
                [self._batch_size, 1])
            x_values = tf.expand_dims(x_values, axis=-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            minval = 0 if self.include_context else 3
            maxval = self._max_num_context - num_context if self.include_context else self._max_num_context
            num_target = tf.random_uniform(
                shape=(), minval=minval, maxval=maxval, dtype=tf.int32)
            num_total_points = num_context + num_target
            x_values = tf.random_uniform(
                [self._batch_size, num_total_points, self._x_size], -20, 20)

        # Set kernel parameters
        l1_scale = tf.random_uniform([], 0.1, 0.6) if self.random_kernel else self._l1_scale
        l1 = (
                tf.ones(shape=[self._batch_size, self._y_size, self._x_size]) *
                l1_scale)
        sigma_scale = tf.random_uniform([], 0.1, 1) if self.random_kernel else self._sigma_scale
        sigma_f = tf.ones(
            shape=[self._batch_size, self._y_size]) * sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = tf.matmul(
            cholesky,
            tf.random_normal([self._batch_size, self._y_size, num_total_points, 1]))

        # [batch_size, num_total_points, y_size]
        y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

        if self._testing:
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
