import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_placeholder():
    z_0 = tf.placeholder(tf.float32, shape=[None, 2])
    log_q_0 = tf.placeholder(tf.float32, shape=[None])
    return z_0, log_q_0


def calc_loss(z_k, log_q_k, target_density):
    log_p = tf.log(target_density.calc_prob_tf(z_k)+1e-7)
    loss = tf.reduce_mean(log_q_k - log_p, -1)
    return loss


def get_train(loss):
    return tf.train.AdamOptimizer().minimize(loss)


class PlanarFlow:
    def __init__(self, dim):
        self.dim = dim
        self.h = lambda x: tf.tanh(x)
        self.h_prime = lambda x: 1 - tf.tanh(x)**2
        self.w = tf.Variable(tf.random.truncated_normal(shape=(1, self.dim)))
        self.b = tf.Variable(tf.zeros(shape=(1)))
        self.u = tf.Variable(tf.random.truncated_normal(shape=(1, self.dim)))
        

    def __call__(self, z, log_q):
        z = z + self.u*self.h(tf.expand_dims(tf.reduce_sum(z*self.w, -1), -1) + self.b)
        psi = self.h_prime(tf.expand_dims(tf.reduce_sum(z*self.w, -1), -1) + self.b)*self.w
        det_jacob = tf.abs(1 + tf.reduce_sum(psi*self.u, -1))
        log_q = log_q -  tf.log(1e-7 + det_jacob)
        return z, log_q


class NormalizingFlow:
    def __init__(self, K, dim):
        self.K = K
        self.dim = dim
        self.planar_flow = [PlanarFlow(self.dim) for i in range(self.K)]
        
        
    def __call__(self, z_0, log_q_0):
        z, log_q = self.planar_flow[0](z_0, log_q_0)
        for pf in self.planar_flow[1:]:
            z, log_q = pf(z, log_q)
        return z, log_q

