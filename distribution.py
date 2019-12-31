import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Distribution:    
    def calc_prob(self, z):
        p = np.zeros(z.shape[0])
        return p
    

    def plot(self, size=5):
        side = np.linspace(-size, size, 1000)
        z1, z2 = np.meshgrid(side, side)
        shape = z1.shape
        z1 = z1.ravel()
        z2 = z2.ravel()
        z = np.c_[z1, z2]
        probability = self.calc_prob(z).reshape(shape)
        plt.figure(figsize=(6, 6))
        plt.imshow(probability)
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        plt.show()


class NormalDistribution2D(Distribution):
    def sample(self, sample_num):
        z = np.random.randn(sample_num, 2)
        return z
    

    def sample_tf(self, sample_num):
        z = tf.random_normal([sample_num, 2])
        return z
    

    def calc_prob(self, z):
        p = np.exp(-(z[:, 0]**2+z[:, 1]**2)/2)/(2*np.pi) 
        return p
    

    def calc_prob_tf(self, z):
        p = tf.exp(-(z[:, 0]**2+z[:, 1]**2)/2)/(2*np.pi) 
        return p


class TargetDistribution1(Distribution):
    def calc_prob(self, z):
        z1, z2 = z[:, 0], z[:, 1]
        norm = np.sqrt(z1**2+z2**2)
        exp1 = np.exp(-0.5*((z1-2)/0.6)**2)
        exp2 = np.exp(-0.5*((z1+2)/0.6)**2)
        p = 0.5*((norm - 2)/0.4)**2 - np.log(exp1 + exp2)
        return np.exp(-p)
    

    def calc_prob_tf(self, z):
        z1, z2 = z[:, 0], z[:, 1]
        norm = tf.sqrt(z1**2+z2**2)
        exp1 = tf.exp(-0.5*((z1-2)/0.6)**2)
        exp2 = tf.exp(-0.5*((z1+2)/0.6)**2)
        p = 0.5*((norm - 2)/0.4)**2 - tf.log(exp1 + exp2)
        return tf.exp(-p)


class TargetDistribution2(Distribution):
    def calc_prob(self, z):
        z1, z2 = z[:, 0], z[:, 1]
        w1 = np.sin(0.5*np.pi*z1)
        p = 0.5*((z2 - w1)/0.4)**2
        return np.exp(-p)
    
    
    def calc_prob_tf(self, z):
        z1, z2 = z[:, 0], z[:, 1]
        w1 = tf.sin(0.5*np.pi*z1)
        p = 0.5*((z2 - w1)/0.4)**2
        return tf.exp(-p)