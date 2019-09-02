
import tensorflow as tf, numpy as np

def dot_product_attention(Q, K, V):
    """
    :param Q: [B x N x D1], K: [B x M x D1], V: [B x M x D2]
    :return: weighted_sum_with_attention: [B x N x D2]
    """
    return tf.matmul(
        tf.nn.softmax(tf.matmul(Q, K, transpose_b=True)/tf.sqrt(tf.cast(tf.shape(Q)[2], Q.dtype))),V)

def multihead_attention(Q_base, K_base, V_base, number_heads, name):
    """
    :param Q: [B x N x D1], K: [B x M x D1], V: [B x M x D2]
    :return: weighted_sum_with_attention: [B x N x D2]
    """
    B, D1, D2 = tf.shape(Q_base)[0], Q_base.shape.as_list()[2], V_base.shape.as_list()[2]
    Dk, Dv = D1//number_heads, D2//number_heads
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        heads = []
        for i in range(number_heads):
            Q = tf.layers.dense(Q_base, Dk, name='WQ_{}'.format(i))
            K = tf.layers.dense(K_base, Dk, name='WK_{}'.format(i))
            V = tf.layers.dense(V_base, Dv, name='WV_{}'.format(i))
            heads.append(dot_product_attention(Q, K, V))
        head = tf.concat(heads, axis=2)
        O = tf.layers.dense(head, D2, name='WO')
    return O

def self_attention_layer(x, number_heads, prefix, suffix):
    """
    :param x: [B, N, D]
    :return:
    """
    name = '{}_self_attention_layer_{}'.format(prefix, suffix)
    input_dim = x.shape.as_list()[2]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        residual = multihead_attention(x, x, x, number_heads, 'multihead_attention_{}'.format(suffix))
        x = tf.contrib.layers.layer_norm(x + residual)
        h = tf.layers.dense(x, input_dim * 2, tf.nn.relu)
        residual = tf.layers.dense(h, input_dim)
    return tf.contrib.layers.layer_norm(x + residual)
