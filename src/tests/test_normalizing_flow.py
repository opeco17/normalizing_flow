import unittest
from unittest import mock

import tensorflow as tf
from tensorflow.python.framework.ops import Operation, Tensor
from tensorflow.python.framework.tensor_shape import TensorShape

from distribution import TargetDistribution1
from normalizing_flow import Flow
from normalizing_flow import NormalizingFlow


class TestFlow(unittest.TestCase):
    
    def setUp(self):
        self.dim = 2
        self.z = tf.random_normal([100, self.dim])
        self.log_q = tf.random.normal([100])
        
    def test_constructor(self):
        with self.assertRaises(NotImplementedError):
            Flow(self.dim)
          
    @mock.patch('normalizing_flow.Flow.__init__')  
    def test_forward(self, mock):
        mock.return_value = None
        flow = Flow(self.dim)
        with self.assertRaises(NotImplementedError):
            flow.forward(self.z, self.log_q)
        

class TestNormalizingFlow(unittest.TestCase):
    
    def setUp(self):
        self.K = 16
        self.dim = 2
        
        self.z = tf.random_normal([100, self.dim])
        self.log_q = tf.random.normal([100])
    
    def test_constructor(self):
        normalizing_flow = NormalizingFlow(self.K, self.dim)
        
    def test_forward(self):
        normalizing_flow = NormalizingFlow(self.K, self.dim)
        z, log_q = normalizing_flow.forward(self.z, self.log_q)
        
        self.assertIsInstance(z, Tensor)
        self.assertEqual(z.shape, TensorShape([100, self.dim]))
        
        self.assertIsInstance(log_q, Tensor)
        self.assertEqual(log_q.shape, TensorShape([100]))
        
    def test_get_placeholder(self):
        normalizing_flow = NormalizingFlow(self.K, self.dim)
        z_0, log_q_0 = normalizing_flow.get_placeholder()
        self.assertIsInstance(z_0, Tensor)
        self.assertEqual(len(z_0.shape), 2)
        
        self.assertIsInstance(log_q_0, Tensor)
        self.assertEqual(len(log_q_0.shape), 1)
        
    def test_calc_loss(self):
        normalizing_flow = NormalizingFlow(self.K, self.dim)
        loss = normalizing_flow.calc_loss(self.z, self.log_q, TargetDistribution1)
        self.assertIsInstance(loss, Tensor)
        self.assertEqual(loss.shape, TensorShape([]))
        
    def test_get_trainer(self):
        normalizing_flow = NormalizingFlow(self.K, self.dim)
        z_0, log_q_0 = normalizing_flow.get_placeholder()
        z, log_q = normalizing_flow.forward(z_0, log_q_0)
        loss = normalizing_flow.calc_loss(z, log_q, TargetDistribution1)
        trainer = normalizing_flow.get_trainer(loss)
        self.assertIsInstance(trainer, Operation)