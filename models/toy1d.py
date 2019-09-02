import numpy as np
import tensorflow as tf
from models.ops import *
from models.Attention import self_attention_layer, multihead_attention, dot_product_attention

import tensorflow_probability as tfp

lr = 5e-5
beta1 = 0.5
beta2 = 0.999

dtype = tf.float32
jitter = 1e-3 if dtype == tf.float64 else 1e-1

class ResidualNeuralProcess(object):
    def __init__(self, model_type, data, z_dim, num_layers=3, is_feature_shared = True):
        assert model_type in ['BLL', 'cANP', 'RNP_full', 'RNP_novar', 'RNP_nomean', 'RNP_notx',
                              'RNP_mean', 'RNP_var', 'RNP_tx', 'NP']

        self.context_x, self.context_y, self.target_x, self.target_y = [tf.cast(d, dtype) for d in data]
        self.model_type = model_type
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.is_feature_shared = is_feature_shared
        self.decoder_size = (self.z_dim, self.z_dim, self.z_dim, self.z_dim, 2)
        self.feature_extractor_size = [self.z_dim * 2] * (num_layers - 1) + [self.z_dim]
        sigma_unconstrained = tf.Variable(-20, name='sigma_unconstrained', dtype=dtype)
        self.out_var = jitter ** 2 + tf.nn.softplus(sigma_unconstrained)

        self.aggregater = 'average' if model_type == 'NP' else 'cross_attention'

        if model_type == 'BLL':
            predictive_dist = self.attention_pseudo_inverse(self.target_x, self.context_x, self.context_y, self.z_dim)
            c_p_dist = self.attention_pseudo_inverse(self.context_x, self.context_x, self.context_y, self.z_dim)
        else:
            r = self.attention_deterministic_path(
                self.target_x, self.context_x, self.context_y, self.z_dim)
            predictive_dist = self.ANP_decoder(self.target_x, r=r, learn_ob_var = True)
            r_c = self.attention_deterministic_path(
                self.context_x, self.context_x, self.context_y, self.z_dim)
            c_p_dist = self.ANP_decoder(self.context_x, r=r_c, learn_ob_var=True)

#        self.NLL = - tf.reduce_mean(predictive_dist.log_prob(self.target_y))
#        self.context_NLL = - tf.reduce_mean(c_p_dist.log_prob(self.context_y))

        NLL = - tf.reduce_mean(predictive_dist.log_prob(self.target_y))
        self.context_NLL = NLL
        self.NLL = tf.losses.mean_squared_error(predictive_dist.mean(), self.target_y)

        with tf.control_dependencies([tf.check_numerics(NLL, 'NLL')]):
            self.optims = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(NLL)

        print('----- Trainable Variables -----')
        for var in tf.trainable_variables():
            print(var.name, var.shape.as_list())
        print('----- Trainable Variables -----')

        self.predict = self.predict_BLL if model_type == 'BLL' else self.predict_cANP


    def feature_extractor(self, x, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for size in self.feature_extractor_size[:-1]:
                x = tf.layers.dense(x, size, tf.nn.relu)
            return tf.layers.dense(x, self.feature_extractor_size[-1], tf.nn.sigmoid)

    def attention_pseudo_inverse(self, tx, cx, cy, z_dim):
        with tf.variable_scope('attention_pinv', reuse=tf.AUTO_REUSE):
            B, NC = tf.shape(cx)[0], tf.shape(cx)[1]
            dy = tf.shape(cy)[2]
            ch = self.feature_extractor(cx, scope='feature_pinv')
            th = self.feature_extractor(tx, scope='feature_pinv')

            Kmm = tf.matmul(ch, ch, transpose_b = True) \
                  + self.out_var * tf.tile(tf.eye(NC, dtype=dtype)[None], [B, 1, 1])

            Lm = tf.cholesky(Kmm)  # B x C x C
            Kmn = tf.matmul(ch, th, transpose_b=True)
            A = tf.matrix_triangular_solve(Lm, Kmn, lower=True) # B x C x T

            Knn = tf.reduce_sum(tf.square(th), axis=2)  # B x T

            fvar = Knn - tf.reduce_sum(tf.square(A), 1) # B x T
            fvar = tf.tile(fvar[:, :, None], [1, 1, dy]) # B x T x Dy

            A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False) # B x C x T
            fmean = tf.matmul(A, cy, transpose_a=True) # B x T x Dy
        return tfp.distributions.MultivariateNormalDiag(fmean, tf.sqrt(fvar) + 1e-4)


    def attention_deterministic_path(self, tx, cx, cy, z_dim):
        with tf.variable_scope('attention_det', reuse=tf.AUTO_REUSE):
            h = tf.concat([cx, cy], axis=2)
            for _ in range(2):
                h = tf.layers.dense(h, z_dim, tf.nn.relu)
            h = tf.layers.dense(h, z_dim)
            for i, head in enumerate((8,8)):
                h = self_attention_layer(h, head, i, 'det')

            ch = self.feature_extractor(cx, 'keyquery_det')
            th = self.feature_extractor(tx, 'keyquery_det')
            if 'RNP' in self.model_type:
                ch2 = ch if self.is_feature_shared else self.feature_extractor(cx, 'keyquery_det2')
                th2 = th if self.is_feature_shared else self.feature_extractor(tx, 'keyquery_det2')

                Kmm = tf.matmul(ch2, ch2, transpose_b=True) \
                      + self.out_var * tf.tile(tf.eye(tf.shape(cx)[1], dtype=dtype)[None], [tf.shape(cx)[0], 1, 1])
                Lm = tf.cholesky(Kmm)  # B x C x C
                thch = tf.matmul(ch2, th2, transpose_b=True) # B x C x T
                A = tf.matrix_triangular_solve(Lm, thch, lower=True) # B x C x T
                A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False) # B x C x T
                r1 = tf.matmul(A, cy, transpose_a=True)
                r2 = th2 - tf.matmul(A, ch2, transpose_a=True)

            if self.aggregater == 'cross_attention':
                r = multihead_attention(th, ch, h, 8, 'cross_attention')
                q = tf.contrib.layers.layer_norm(r + th)
                qh = tf.layers.dense(q, z_dim * 2, tf.nn.relu)
                r = tf.contrib.layers.layer_norm(tf.layers.dense(qh, z_dim) + q)
            else:
                r = tf.tile(tf.reduce_mean(h, axis=1, keepdims=True), [1, tf.shape(tx)[1], 1])
            if 'RNP' in self.model_type:
                if 'full' in self.model_type:
                    r = tf.concat([r, r1, r2, th2], axis=2)
                elif 'nomean' in self.model_type:
                    r = tf.concat([r, r2, th2], axis=2)
                elif 'novar' in self.model_type:
                    r = tf.concat([r, r1, th2], axis=2)
                elif 'notx' in self.model_type:
                    r = tf.concat([r, r1, r2], axis=2)
                elif '_mean' in self.model_type:
                    r = tf.concat([r, r1], axis=2)
                elif '_var' in self.model_type:
                    r = tf.concat([r, r2], axis=2)
                elif '_tx' in self.model_type:
                    r = tf.concat([r, th2], axis=2)
        return r

    def ANP_decoder(self, target_x, r=None, z=None, learn_ob_var = False):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            num_target = tf.shape(target_x)[1]
            with tf.control_dependencies([num_target]):
                h = target_x
                if r is not None:
                    h = tf.concat([h, r], axis=2)
                if z is not None:
                    z_tiled = tf.tile(z, [1, num_target, 1])
                    h = tf.concat([h, z_tiled], axis=2)

                for size in self.decoder_size[:-1]:
                    h = tf.layers.dense(h, size, tf.nn.relu)
                x = tf.layers.dense(h, self.decoder_size[-1])
                mu, log_sigma = tf.split(x, 2, -1)
                if learn_ob_var:
                    sigma = jitter + tf.nn.softplus(log_sigma)
                else:
                    sigma = 0.05 * tf.ones_like(mu)
        return tfp.distributions.MultivariateNormalDiag(mu, sigma)

    def predict_BLL(self, data, number_of_samples):
        context_x, context_y, target_x, _ = [tf.cast(d, dtype) for d in data]
        predictive_dist = self.attention_pseudo_inverse(target_x, context_x, context_y, self.z_dim)
        return predictive_dist.mean(), predictive_dist.stddev()

    def predict_cANP(self, data, number_of_samples):
        context_x, context_y, target_x, _ = [tf.cast(d, dtype) for d in data]
        r = self.attention_deterministic_path(target_x, context_x, context_y, self.z_dim)
        predictive_dist = self.ANP_decoder(target_x, r=r, learn_ob_var = True)
        return predictive_dist.mean(), predictive_dist.stddev()
