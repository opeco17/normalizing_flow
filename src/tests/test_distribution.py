import unittest
from unittest import mock

import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.tensor_shape import TensorShape

from distribution import Distribution
from distribution import NormalDistribution
from distribution import TargetDistribution1
from distribution import TargetDistribution2


class TestDistribution(unittest.TestCase):
    
    def setUp(self):
        self.z = np.random.randn(100, 2)
        self.z_tf = z = tf.random_normal([100, 2])
    
    def test_calc_prob(self):
        with self.assertRaises(NotImplementedError):
            Distribution.calc_prob(self.z)
            
    def test_calc_prob_tf(self):
        with self.assertRaises(NotImplementedError):
            Distribution.calc_prob_tf(self.z_tf)
          
    def test_save_distribution(self):
        with self.assertRaises(NotImplementedError):
            Distribution.save_distribution()
            
            
class TestNormalDistribution(unittest.TestCase):
    
    def setUp(self):
        self.sample_num = 100
        self.dim = 2
        self.z = np.random.randn(100, self.dim)
        self.z_tf = z = tf.random_normal([100, self.dim])
        
    def test_sample(self):
        z = NormalDistribution.sample(self.sample_num)
        self.assertIsInstance(z, ndarray)
        self.assertEqual(z.shape, (self.sample_num, self.dim))
        
    def test_sample_tf(self):
        z = NormalDistribution.sample_tf(self.sample_num)
        self.assertIsInstance(z, Tensor)
        self.assertEqual(z.shape, TensorShape([self.sample_num, self.dim]))
    
    def test_calc_prob(self):
        p = NormalDistribution.calc_prob(self.z)
        self.assertIsInstance(p, ndarray)
        self.assertEqual(p.shape, (100, ))
        
    def test_calc_prob_tf(self):
        p = NormalDistribution.calc_prob_tf(self.z)
        self.assertIsInstance(p, Tensor)
        self.assertEqual(p.shape, TensorShape([100]))
        
    @mock.patch('distribution.plt.savefig')
    def test_save_distribution(self, mock):
        mock.return_value = None
        NormalDistribution.save_distribution()
        
        
class TestTargetDistribution1(unittest.TestCase):
    
    def setUp(self):
        self.sample_num = 100
        self.dim = 2
        self.z = np.random.randn(100, self.dim)
        self.z_tf = z = tf.random_normal([100, self.dim])
    
    def test_calc_prob(self):
        p = TargetDistribution1.calc_prob(self.z)
        self.assertIsInstance(p, ndarray)
        self.assertEqual(p.shape, (100, ))
        
    def test_calc_prob_tf(self):
        p = TargetDistribution1.calc_prob_tf(self.z)
        self.assertIsInstance(p, Tensor)
        self.assertEqual(p.shape, TensorShape([100]))
        
    @mock.patch('distribution.plt.savefig')
    def test_save_distribution(self, mock):
        mock.return_value = None
        TargetDistribution1.save_distribution()
        
        
class TestTargetDistribution2(unittest.TestCase):
    
    def setUp(self):
        self.sample_num = 100
        self.dim = 2
        self.z = np.random.randn(100, self.dim)
        self.z_tf = z = tf.random_normal([100, self.dim])
    
    def test_calc_prob(self):
        p = TargetDistribution2.calc_prob(self.z)
        self.assertIsInstance(p, ndarray)
        self.assertEqual(p.shape, (100, ))
        
    def test_calc_prob_tf(self):
        p = TargetDistribution2.calc_prob_tf(self.z)
        self.assertIsInstance(p, Tensor)
        self.assertEqual(p.shape, TensorShape([100]))
        
    @mock.patch('distribution.plt.savefig')
    def test_save_distribution(self, mock):
        mock.return_value = None
        TargetDistribution2.save_distribution()