import os

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import tensorflow as tf

from distribution import *
from normalizing_flow import *


def show_result(iteration: int, z_k_value: ndarray) -> None:
    """Show samples from Normalizing Flow"""
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    plt.scatter(z_k_value[:, 0], z_k_value[:, 1], alpha=0.6)
    if AXIS_INVISIBLE:
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.savefig(os.path.join(FIGURE_DIR, f'result_{iteration}iteration.png'))
    plt.show()
                
                
def train(TargetDistribution: Distribution) -> None:
    """Training Normalizing Flow"""
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
        
    normalizing_flow = NormalizingFlow(K=16, dim=2)

    z_0, log_q_0 = get_placeholder()
    z_k, log_q_k = normalizing_flow(z_0, log_q_0)
    loss = calc_loss(z_k, log_q_k, TargetDistribution)
    train = get_train(loss)

    with tf.Session() as sess:
        invisible_axis = True
        sess.run(tf.global_variables_initializer())

        for iteration in range(ITERATION+1):
            z_0_batch = NormalDistribution.sample(BATCH_SIZE)
            log_q_0_batch = np.log(NormalDistribution.calc_prob(z_0_batch))
            _, loss_value = sess.run([train, loss], {z_0:z_0_batch, log_q_0:log_q_0_batch})

            if iteration % 100 == 0:
                print(f'Iteration : {iteration}   Loss : {loss_value}')

            if iteration % SAVE_FIGURE_INTERVAL == 0:
                z_0_batch_for_visualize = NormalDistribution.sample(NUMBER_OF_SAMPLES_FOR_VISUALIZE)
                log_q_0_batch_for_visualize = np.log(NormalDistribution.calc_prob(z_0_batch_for_visualize))
                z_k_value = sess.run(z_k, {z_0:z_0_batch_for_visualize, log_q_0:log_q_0_batch_for_visualize})
                show_result(iteration, z_k_value)


if __name__ == '__main__':
    train(TargetDistribution1)
    train(TargetDistribution2)