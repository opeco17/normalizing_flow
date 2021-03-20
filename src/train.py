import argparse
import os

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

from settings import *
from distribution import Distribution, NormalDistribution, TargetDistribution1, TargetDistribution2
from logger import logger
from normalizing_flow import NormalizingFlow
from utils import save_loss_values, save_result
                
                
def train(target_distribution: Distribution) -> None:
    """Training Normalizing Flow"""
    
    target_distribution.save_distribution()
        
    normalizing_flow = NormalizingFlow(K=NORMALIZING_FLOW_LAYER_NUM)

    z_0, log_q_0 = normalizing_flow.get_placeholder()
    z_k, log_q_k = normalizing_flow.forward(z_0, log_q_0)
    loss = normalizing_flow.calc_loss(z_k, log_q_k, target_distribution)
    trainer = normalizing_flow.get_trainer(loss)
    logger.info('Calculation graph constructed')

    loss_values = []
    
    with tf.Session() as sess:
        logger.info('Session Start')
        sess.run(tf.global_variables_initializer())
        logger.info('All variables initialized')
        logger.info(f'Training Start (number of iterations: {ITERATION})')

        for iteration in range(ITERATION+1):
            z_0_batch = NormalDistribution.sample(BATCH_SIZE)
            log_q_0_batch = np.log(NormalDistribution.calc_prob(z_0_batch))
            _, loss_value = sess.run([trainer, loss], {z_0:z_0_batch, log_q_0:log_q_0_batch})
            loss_values.append(loss_value)

            if iteration % 100 == 0:
                iteration_digits = len(str(ITERATION))
                logger.info(f'Iteration:  {iteration:<{iteration_digits}}  Loss:  {loss_value}')

            if iteration % SAVE_FIGURE_INTERVAL == 0:
                z_0_batch_for_visualize = NormalDistribution.sample(NUMBER_OF_SAMPLES_FOR_VISUALIZE)
                log_q_0_batch_for_visualize = np.log(NormalDistribution.calc_prob(z_0_batch_for_visualize))
                z_k_value = sess.run(z_k, {z_0:z_0_batch_for_visualize, log_q_0:log_q_0_batch_for_visualize})
                save_result(z_k_value, iteration, target_distribution.__name__)
                save_loss_values(loss_values, target_distribution.__name__)
                
        logger.info('Training Finished')
        
    logger.info('Session Closed')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--target_distribution', type=int)
    args = parser.parse_args()

    if args.target_distribution == 1:
        train(TargetDistribution1)
    elif args.target_distribution == 2:
        train(TargetDistribution2)
    else:
        train(TargetDistribution1)
        train(TargetDistribution2)