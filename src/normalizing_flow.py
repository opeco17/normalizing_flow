from abc import abstractclassmethod
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import Operation, Tensor

from settings import *
from distribution import Distribution


class Flow:
    
    @abstractclassmethod
    def __init__(self, dim: int) -> None:
        raise NotImplementedError
        
    @abstractclassmethod
    def forward(self, z: Tensor, log_q: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
    

class PlanarFlow(Flow):
    
    def __init__(self, dim: int=2) -> None:
        self.dim = dim
        self.h = lambda x: tf.tanh(x)
        self.h_prime = lambda x: 1 - tf.tanh(x) ** 2
        self.w = tf.Variable(tf.random.truncated_normal(shape=(1, self.dim)))
        self.b = tf.Variable(tf.zeros(shape=(1)))
        self.u = tf.Variable(tf.random.truncated_normal(shape=(1, self.dim)))

    def forward(self, z: Tensor, log_q: Tensor) -> Tuple[Tensor, Tensor]:
        z = z + self.u * self.h(tf.expand_dims(tf.reduce_sum(z * self.w, -1), -1) + self.b)
        psi = self.h_prime(tf.expand_dims(tf.reduce_sum(z * self.w, -1), -1) + self.b) * self.w
        det_jacob = tf.abs(1 + tf.reduce_sum(psi * self.u, -1))
        log_q = log_q -  tf.log(det_jacob + DELTA)
        return z, log_q


class NormalizingFlow:
    
    def __init__(self, K: int=16, dim: int=2) -> None:
        self.K = K
        self.dim = dim
        self.planar_flows = [PlanarFlow(self.dim) for i in range(self.K)]
        
    def forward(self, z_0: Tensor, log_q_0: Tensor) -> Tuple[Tensor, Tensor]:
        z, log_q = self.planar_flows[0].forward(z_0, log_q_0)
        for planar_flow in self.planar_flows[1:]:
            z, log_q = planar_flow.forward(z, log_q)
        return z, log_q

    def get_placeholder(self) -> Tuple[Tensor, Tensor]:
        z_0 = tf.placeholder(tf.float32, shape=[None, self.dim])
        log_q_0 = tf.placeholder(tf.float32, shape=[None])
        return z_0, log_q_0

    def calc_loss(self, z_k: Tensor, log_q_k: Tensor, target_distribution: Distribution) -> Tensor:
        p = target_distribution.calc_prob_tf(z_k)
        log_p = tf.log(p + DELTA)
        loss = tf.reduce_mean(log_q_k - log_p)
        return loss

    def get_trainer(self, loss: Tensor) -> Operation:
        trainer =  tf.train.AdamOptimizer(LEARNING_RATE, BETA1, BETA2).minimize(loss)
        return trainer