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
    def __init__(self, model_type, data, z_dim, num_layers=4, include_context=False):
        assert model_type in ['BLL', 'ANP' ,'fANP', 'bANP', 'RNP', 'fRNP', 'bRNP']

        self.f_dim = 5
        self.context_x, self.context_y, self.target_x, self.target_y = [tf.cast(d, dtype) for d in data]

        k = 50
        tiled_context_x, tiled_context_y, tiled_target_x, tiled_target_y = \
            [tf.tile(tf.cast(d, dtype), [k, 1, 1]) for d in data]
        num_context = tf.shape(self.context_x)[1]
        if include_context:
            unseen_target_x, unseen_target_y = tiled_target_x[:, num_context:], tiled_target_y[:, num_context:]
        else:
            unseen_target_x, unseen_target_y = tiled_target_x, tiled_target_y

        self.model_type = model_type
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.decoder_size = (self.z_dim, self.z_dim, self.z_dim, self.z_dim, 2)
        self.feature_extractor_size = [self.z_dim] * num_layers
        sigma_unconstrained = tf.Variable(-20, name='sigma_unconstrained', dtype=dtype)
        self.out_var = jitter ** 2 + tf.nn.softplus(sigma_unconstrained)

        self.aggregater = 'average' if model_type == 'NP' else 'cross_attention'

        z_c = self.self_attention_latent_path(self.context_x, self.context_y, self.z_dim, 'z_latent')
        if include_context:
            z_t = self.self_attention_latent_path(self.target_x, self.target_y, self.z_dim, 'z_latent')
        else:
            z_t = self.self_attention_latent_path(
                tf.concat([self.context_x, self.target_x], axis=1),
                tf.concat([self.context_y, self.target_y], axis=1), self.z_dim, 'z_latent')
        z_t_samples = None if model_type in ['BLL', 'fANP', 'fRNP'] else z_t.sample()
        z_c_samples = None
        if model_type not in ['BLL', 'fANP', 'fRNP']:
            z_c_samples = z_c.sample(k)
            z_shape = tf.shape(z_c_samples)
            z_c_samples = tf.reshape(z_c_samples, [k * z_shape[1], 1, z_shape[3]])
            z_c_samples.set_shape([None, 1, self.z_dim])

        f_c = self.self_attention_latent_path(self.context_x, self.context_y, self.f_dim, 'f_latent')
        if include_context:
            f_t = self.self_attention_latent_path(self.target_x, self.target_y, self.f_dim, 'f_latent')
        else:
            f_t = self.self_attention_latent_path(
                tf.concat([self.context_x, self.target_x], axis=1),
                tf.concat([self.context_y, self.target_y], axis=1), self.f_dim, 'f_latent')
        f_t_samples = None if model_type in ['ANP', 'RNP'] else f_t.sample()
        f_c_samples = None
        if model_type not in ['ANP', 'RNP']:
            f_c_samples = f_c.sample(k)
            f_shape = tf.shape(f_c_samples)
            f_c_samples = tf.reshape(f_c_samples, [k * f_shape[1], 1, f_shape[3]])
            f_c_samples.set_shape([None, 1, self.f_dim])

        if model_type == 'BLL':
            reconstruction = self.attention_pseudo_inverse(self.target_x, self.context_x, self.context_y, f_t_samples)

            self.ZKL = tf.zeros([])
            self.ELBO= tf.reduce_sum(reconstruction.log_prob(self.target_y), axis=1, keepdims=True)
            self.FKL = tf.reduce_mean(tfp.distributions.kl_divergence(f_t,f_c))
            self.ELBO = self.ELBO - tfp.distributions.kl_divergence(f_t,f_c)
            self.ELBO = tf.reduce_mean(self.ELBO)

            p_dist = self.attention_pseudo_inverse(unseen_target_x, tiled_context_x, tiled_context_y, f_c_samples)
            cp_dist = self.attention_pseudo_inverse(tiled_context_x, tiled_context_x, tiled_context_y, f_c_samples)
        else:
            r = self.attention_deterministic_path(
                self.target_x, self.context_x, self.context_y, self.z_dim, f_t_samples)

            reconstruction = self.ANP_decoder(self.target_x, r=r, z=z_t_samples)

            self.ZKL, self.FKL = tf.zeros([]), tf.zeros([])
            self.ELBO= tf.reduce_sum(reconstruction.log_prob(self.target_y), axis=1, keepdims=True)
            if self.model_type in ['ANP', 'bANP', 'RNP', 'bRNP']:
                self.ZKL = tf.reduce_mean(tfp.distributions.kl_divergence(z_t,z_c))
                self.ELBO = self.ELBO - tfp.distributions.kl_divergence(z_t,z_c)
            if self.model_type in ['fANP', 'bANP', 'fRNP', 'bRNP']:
                self.FKL = tf.reduce_mean(tfp.distributions.kl_divergence(f_t,f_c))
                self.ELBO = self.ELBO - tfp.distributions.kl_divergence(f_t,f_c)
            self.ELBO = tf.reduce_mean(self.ELBO)

            r_t = self.attention_deterministic_path(
                unseen_target_x, tiled_context_x, tiled_context_y, self.z_dim, f_c_samples)
            r_c = self.attention_deterministic_path(
                tiled_context_x, tiled_context_x, tiled_context_y, self.z_dim, f_c_samples)
            p_dist = self.ANP_decoder(unseen_target_x, r=r_t, z=z_c_samples)
            cp_dist = self.ANP_decoder(tiled_context_x, r=r_c, z=z_c_samples)

        self.NLL = p_dist.log_prob(unseen_target_y)
        rep_shape = tf.shape(self.NLL)
        self.NLL = tf.reshape(self.NLL, [k, rep_shape[0]//k, rep_shape[1]])
        self.NLL = tf.reduce_logsumexp(self.NLL, axis=0) - tf.log(tf.cast(k, dtype))
        self.NLL = - tf.reduce_mean(self.NLL)

        self.context_NLL = cp_dist.log_prob(tiled_context_y)
        rep_shape = tf.shape(self.context_NLL)
        self.context_NLL = tf.reshape(self.context_NLL, [k, rep_shape[0]//k, rep_shape[1]])
        self.context_NLL = tf.reduce_logsumexp(self.context_NLL, axis=0) - tf.log(tf.cast(k, dtype))
        self.context_NLL = - tf.reduce_mean(self.context_NLL)

        with tf.control_dependencies([tf.check_numerics(self.ELBO, 'ELBO')]):
            self.optims = tf.train.AdamOptimizer(lr).minimize(-self.ELBO)

        print('----- Trainable Variables -----')
        for var in tf.trainable_variables():
            print(var.name, var.shape.as_list())
        print('----- Trainable Variables -----')

        self.predict = self.predict_BLL if model_type == 'BLL' else self.predict_ANP


    def feature_extractor(self, x, f=None, scope='feature_extractor'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if f is not None:
                f_tiled = tf.tile(f, [1, tf.shape(x)[1], 1])
                x = tf.concat([x, f_tiled], axis=2)
            for size in self.feature_extractor_size[:-1]:
                x = tf.layers.dense(x, size, tf.nn.relu)
            return tf.layers.dense(x, self.feature_extractor_size[-1], tf.nn.sigmoid)

    def attention_pseudo_inverse(self, tx, cx, cy, f):
        with tf.variable_scope('attention_pinv', reuse=tf.AUTO_REUSE):
            B, NC = tf.shape(cx)[0], tf.shape(cx)[1]
            dy = tf.shape(cy)[2]
            ch = self.feature_extractor(cx, f, scope='feature_pinv')
            th = self.feature_extractor(tx, f, scope='feature_pinv')

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


    def self_attention_latent_path(self, x, y, output_dim, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = tf.concat([x, y], axis=2)
            for _ in range(2):
                h = tf.layers.dense(h, self.z_dim, tf.nn.relu)
            h = tf.layers.dense(h, self.z_dim)

            for i, head in enumerate((8,8)):
                h = self_attention_layer(h, head, i, 'lat')
            h = tf.reduce_mean(h, axis=1, keepdims=True)
            h = tf.layers.dense(h, self.z_dim, tf.nn.relu)
            h = tf.layers.dense(h, output_dim * 2)
            mu, log_sigma = tf.split(h, 2, axis=-1)
            sigma = 0.1 + 0.9 * tf.nn.sigmoid(log_sigma)
        return tfp.distributions.MultivariateNormalDiag(mu, sigma)

    def attention_deterministic_path(self, tx, cx, cy, z_dim, f):
        with tf.variable_scope('attention_det', reuse=tf.AUTO_REUSE):
            h = tf.concat([cx, cy], axis=2)
            if f is not None:
                f_tiled = tf.tile(f, [1, tf.shape(h)[1], 1])
                h = tf.concat([h, f_tiled], axis=2)
            for _ in range(2):
                h = tf.layers.dense(h, z_dim, tf.nn.relu)
            h = tf.layers.dense(h, z_dim)
            for i, head in enumerate((8,8)):
                h = self_attention_layer(h, head, i, 'det')

            ch = self.feature_extractor(cx, f, scope='keyquery_det')
            th = self.feature_extractor(tx, f, scope='keyquery_det')
            if 'RNP' in self.model_type:
                Kmm = tf.matmul(ch, ch, transpose_b=True) \
                      + self.out_var * tf.tile(tf.eye(tf.shape(cx)[1], dtype=dtype)[None], [tf.shape(cx)[0], 1, 1])
                Lm = tf.cholesky(Kmm)  # B x C x C
                thch = tf.matmul(ch, th, transpose_b=True) # B x C x T
                A = tf.matrix_triangular_solve(Lm, thch, lower=True) # B x C x T
                A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False) # B x C x T
                r1 = tf.matmul(A, cy, transpose_a=True)
                r2 = th - tf.matmul(A, ch, transpose_a=True)

            if self.aggregater == 'cross_attention':
                r = multihead_attention(th, ch, h, 8, 'cross_attention')
                q = tf.contrib.layers.layer_norm(r + th)
                qh = tf.layers.dense(q, z_dim * 2, tf.nn.relu)
                r = tf.contrib.layers.layer_norm(tf.layers.dense(qh, z_dim) + q)
            else:
                r = tf.tile(tf.reduce_mean(h, axis=1, keepdims=True), [1, tf.shape(tx)[1], 1])

            if 'RNP' in self.model_type:
                r = tf.concat([r, r1, r2, th], axis=2)
        return r

    def ANP_decoder(self, target_x, r=None, z=None, learn_ob_var = True):
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
        f = self.self_attention_latent_path(context_x, context_y, self.f_dim, 'f_latent')
        sample_f = None if self.model_type in ['ANP', 'RNP'] else f.sample(number_of_samples)
        f_shape = tf.shape(sample_f)
        sample_f = tf.reshape(sample_f, [number_of_samples * f_shape[1], 1, f_shape[3]])
        sample_f.set_shape([None, 1, self.f_dim])

        context_x, context_y, target_x, _ = [tf.tile(tf.cast(d, dtype), [number_of_samples, 1, 1]) for d in data]
        dist = self.attention_pseudo_inverse(target_x, context_x, context_y, sample_f)
        rep_p_mu = dist.mean()
        rep_p_sigma = dist.stddev()
        rep_shape = tf.shape(rep_p_mu)
        mu = tf.reshape(rep_p_mu, [number_of_samples, rep_shape[0]//number_of_samples, rep_shape[1], rep_shape[2]])
        sigma = tf.reshape(rep_p_sigma, [number_of_samples, rep_shape[0]//number_of_samples, rep_shape[1], rep_shape[2]])

        return mu, sigma

    def predict_ANP(self, data, number_of_samples):

        context_x, context_y, target_x, _ = [tf.cast(d, dtype) for d in data]
        z = self.self_attention_latent_path(context_x, context_y, self.z_dim, 'z_latent')
        sample_z = None
        if self.model_type not in ['fANP', 'fRNP']:
            sample_z = z.sample(number_of_samples)
            z_shape = tf.shape(sample_z)
            sample_z = tf.reshape(sample_z, [number_of_samples * z_shape[1], 1, z_shape[3]])
            sample_z.set_shape([None, 1, self.z_dim])

        f = self.self_attention_latent_path(context_x, context_y, self.f_dim, 'f_latent')
        sample_f = None
        if self.model_type not in ['ANP', 'RNP']:
            sample_f = f.sample(number_of_samples)
            f_shape = tf.shape(sample_f)
            sample_f = tf.reshape(sample_f, [number_of_samples * f_shape[1], 1, f_shape[3]])
            sample_f.set_shape([None, 1, self.f_dim])

        context_x, context_y, target_x, _ = [tf.tile(tf.cast(d, dtype), [number_of_samples, 1, 1]) for d in data]
        r = self.attention_deterministic_path(target_x, context_x, context_y, self.z_dim, sample_f)

        dist = self.ANP_decoder(target_x, r=r, z=sample_z)
        rep_p_mu = dist.mean()
        rep_p_sigma = dist.stddev()
        rep_shape = tf.shape(rep_p_mu)
        mu = tf.reshape(rep_p_mu, [number_of_samples, rep_shape[0]//number_of_samples, rep_shape[1], rep_shape[2]])
        sigma = tf.reshape(rep_p_sigma, [number_of_samples, rep_shape[0]//number_of_samples, rep_shape[1], rep_shape[2]])

        return mu, sigma
