import unittest
from unittest import mock

import numpy as np

from utils import camel_to_snake
from utils import save_loss_values
from utils import save_result


class TestCamelToSnake(unittest.TestCase):
    
    def test_camel_to_snake1(self):
        snake_str = 'this_is_string'
        camel_str = 'ThisIsString'
        self.assertEqual(camel_to_snake(camel_str), snake_str)
    
    def test_camel_to_snake2(self):
        snake_str = 'normalizing_flow1'
        camel_str = 'NormalizingFlow1'
        self.assertEqual(camel_to_snake(camel_str), snake_str)
    

class TestSaveLossValues(unittest.TestCase):
    
    def setUp(self):
        self.loss_values = np.arange(10)
        
    @mock.patch('utils.plt.savefig')
    @mock.patch('utils.os.makedirs')
    def test_save_loss_values(self, makedirs_mock, savefig_mock):
        makedirs_mock.return_value = None
        savefig_mock.return_value = None
        
        save_loss_values(self.loss_values, 'distribution_name')
        
        
class TestSaveResult(unittest.TestCase):
    
    def setUp(self):
        self.iteration = 100
        self.z_k_value = np.random.randn(100, 2)
        
    @mock.patch('utils.plt.savefig')
    @mock.patch('utils.os.makedirs')
    def test_save_result(self, makedirs_mock, savefig_mock):
        makedirs_mock.return_value = None
        savefig_mock.return_value = None
        
        save_result(self.z_k_value, self.iteration, 'distribution_name')