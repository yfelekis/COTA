import numpy as np
import pandas as pd
import cvxpy as cp

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from pgmpy import inference

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

def visualize_res(plans, cost, cost_func, show_values, base_labels, abst_labels):
    num_matrices = len(plans)

    num_rows = (num_matrices - 1) // 3 + 1  # Calculate the number of rows needed
    num_cols = min(num_matrices, 3)  # Number of columns is fixed at 3

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6))# * num_rows))
    col = [''.join(map(str, tpl)) for tpl in base_labels]
    row = [''.join(map(str, tpl)) for tpl in abst_labels]
    
    im = axs[0, 0].matshow(plans[0].T, vmin=0, vmax=1)  # Use the first subplot as reference for colorbar size

    for i, pl in enumerate(plans):
        row_idx = i // 3  # Calculate the row index
        col_idx = i % 3   # Calculate the column index
        
        im = axs[row_idx, col_idx].matshow(pl, vmin=0, vmax=1)
        axs[row_idx, col_idx].set_xticks(np.arange(len(col)))
        axs[row_idx, col_idx].set_xticklabels(col, rotation=90, fontsize=8)
        axs[row_idx, col_idx].set_yticks(np.arange(len(row)))
        axs[row_idx, col_idx].set_yticklabels(row, rotation=0, fontsize=8)
        if show_values:
            for (m, n), value in np.ndenumerate(pl):
                axs[row_idx, col_idx].text(n, m, '{:0.2f}'.format(value), ha='center', va='center', fontsize=7, color='w')

    fig.suptitle(f"Cost Function: {cost_func} \n Total Cost = {cost}", fontsize=9)
    cax = fig.add_axes([0.2, 0.001, 0.63, 0.03])  # create axis for colorbar
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.subplots_adjust(wspace=0.3, hspace=0.1, top=0.95, bottom=0.05)

    plt.show()
    return