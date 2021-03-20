from abc import abstractmethod
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

from settings import *
from utils import camel_to_snake


class Distribution:
    
    @classmethod
    @abstractmethod
    def calc_prob(cls, z: ndarray) -> ndarray:
        """Calculate probability of ndarray input samples"""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def calc_prob_tf(cls, z: Tensor) -> Tensor:
        """Calculate probability of Tensor input samples"""
        raise NotImplementedError
        
    @classmethod
    def save_distribution(cls, size: int=5) -> None:
        """Show probability distribution"""
        if not os.path.exists(FIGURE_DIR):
            os.makedirs(FIGURE_DIR)
        side = np.linspace(-size, size, FIGURE_RESOLUTION)
        z1, z2 = np.meshgrid(side, side)
        z1, z2 = z1.ravel(), z2.ravel()
        z = np.c_[z1, z2]
        p = cls.calc_prob(z).reshape((FIGURE_RESOLUTION, FIGURE_RESOLUTION))
        
        plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
        plt.imshow(p)
        plt.tick_params(top=False, bottom=False, left=False, right=False)
        plt.tick_params(labeltop=False, labelbottom=False,labelleft=False,labelright=False)
        plt.savefig(os.path.join(FIGURE_DIR, f'{camel_to_snake(cls.__name__)}.png'))
        plt.clf()
        plt.close()


class NormalDistribution(Distribution):
    
    @classmethod
    def sample(cls, sample_num: int) -> ndarray:
        z = np.random.randn(sample_num, 2)
        return z
    
    @classmethod
    def sample_tf(cls, sample_num: int) -> Tensor:
        z = tf.random_normal([sample_num, 2])
        return z

    @classmethod
    def calc_prob(cls, z: ndarray) -> ndarray:
        squared_norm = np.sum(z ** 2, axis=1)
        p = np.exp(-squared_norm / 2) / (2 * np.pi) 
        return p
    
    @classmethod
    def calc_prob_tf(cls, z):
        squared_norm = tf.reduce_sum(z ** 2, axis=1)
        p = tf.exp(-squared_norm / 2) / (2 * np.pi) 
        return p


class TargetDistribution1(Distribution):
    
    @classmethod
    def calc_prob(cls, z: ndarray) -> ndarray:
        z1 = z[:, 0]
        norm = np.sqrt(np.sum(z ** 2, axis=1))
        exp1 = np.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
        exp2 = np.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
        p = 0.5 * ((norm - 2) / 0.4) ** 2 - np.log(exp1 + exp2)
        return np.exp(-p)
    
    @classmethod
    def calc_prob_tf(cls, z: Tensor) -> Tensor:
        z1 = z[:, 0]
        norm = tf.sqrt(tf.reduce_sum(z ** 2, axis=1))
        exp1 = tf.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
        exp2 = tf.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
        p = 0.5 * ((norm - 2) / 0.4) ** 2 - tf.log(exp1 + exp2)
        return tf.exp(-p)


class TargetDistribution2(Distribution):
    
    @classmethod
    def calc_prob(cls, z: ndarray) -> ndarray:
        z1, z2 = z[:, 0], z[:, 1]
        w1 = np.sin(0.5 * np.pi * z1)
        p = 0.5 * ((z2 - w1) / 0.4) ** 2
        return np.exp(-p)
    
    @classmethod
    def calc_prob_tf(cls, z: Tensor) -> Tensor:
        z1, z2 = z[:, 0], z[:, 1]
        w1 = tf.sin(0.5 * np.pi * z1)
        p = 0.5 * ((z2 - w1) / 0.4) ** 2
        return tf.exp(-p)