import os
import re
from typing import List

import matplotlib.pyplot as plt
from numpy import ndarray

from settings import *


def camel_to_snake(camel_str: str) -> str:
    """Convert camel case string to snake case string"""
    snake_str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    snake_str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_str).lower()
    return snake_str


def save_loss_values(loss_values: List, distribution_name: str, loss_interval=1) -> None:
    result_figure_dir = os.path.join(FIGURE_DIR, f'result_{camel_to_snake(distribution_name)}')
    if not os.path.exists(result_figure_dir):
        os.makedirs(result_figure_dir)
    
    plt.figure(figsize=(6, 4))
    plt.plot([loss_interval * i for i in range(len(loss_values))], loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(result_figure_dir, f'loss.png'))
    plt.clf()
    plt.close()
    

def save_result(z_k_value: ndarray, iteration: int, distribution_name: str) -> None:
    """Save samples from Normalizing Flow"""
    result_figure_dir = os.path.join(FIGURE_DIR, f'result_{camel_to_snake(distribution_name)}')
    if not os.path.exists(result_figure_dir):
        os.makedirs(result_figure_dir)
    
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    plt.scatter(z_k_value[:, 0], z_k_value[:, 1], alpha=0.6)
    if AXIS_INVISIBLE:
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
    plt.savefig(os.path.join(result_figure_dir, f'{iteration}iteration.png'))
    plt.clf()
    plt.close()