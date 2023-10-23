import cvxpy as cp
import itertools
import joblib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import networkx as nx
import numpy as np
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import pandas as pd
import scipy.optimize as optimize
from scipy.spatial.distance import cdist, squareform, pdist
import seaborn as sns
from collections import Counter
from IPython.utils import io
from cvxpy.error import SolverError
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from pgmpy import inference
from scipy.stats import wasserstein_distance
import modularized_utils as ut
from src.examples import smokingmodels as sm
import maps
import classes as cls
from scipy.optimize import linprog
from scipy import stats

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def frobenius_distance(matrix1, matrix2):

    diff             = matrix1 - matrix2
    squared_diff     = np.square(diff)
    sum_squared_diff = np.sum(squared_diff)
    frobenius_dist   = np.sqrt(sum_squared_diff)
    
    return frobenius_dist

def replace_zeros_with_ones(matrix):
    """
    Replaces all zeros in the matrix with ones.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                matrix[i][j] = 1
    return matrix

def renormalize(mat):
    min_val = np.min(mat)
    max_val = np.max(mat)
    norm_mat = (mat - min_val) / (max_val - min_val)
    return norm_mat

def generate_euclidean_cost_matrix(dists):
    
    p = np.arange(1,len(dists[0])+1,1)
    q = np.arange(1,len(dists[1])+1,1)
    p = p.reshape(-1, 1)
    q = q.reshape(-1, 1)
    cost_matrix = cdist(q, p, metric='euclidean')
    
    return cost_matrix 

def generate_sqeuclidean_cost_matrix(dists):
    
    p = np.arange(1,len(dists[0])+1,1)
    q = np.arange(1,len(dists[1])+1,1)
    p = p.reshape(-1, 1)
    q = q.reshape(-1, 1)
    cost_matrix = cdist(q, p, metric='sqeuclidean')
    
    return cost_matrix 

def generate_quadratic2_cost_matrix(dists):
    
    p = np.arange(1, len(dists[0])+1, 1)
    q = np.arange(1, len(dists[1])+1, 1)
    p = p.reshape(-1, 1)
    q = q.reshape(-1, 1)
    
    cost_matrix = cdist(q, p, metric=lambda u, v: 0.5*np.sum(np.square(u-v)))
    
    return cost_matrix 

def generate_omega_cost_matrix(pairs):
    
    lst_base = pairs[0].base_labels
    lst_abst = pairs[0].abst_labels
    df = pd.DataFrame(0, index=lst_abst, columns=lst_base)
    for pair in pairs:
        p = pair.get_domain('base')
        q = pair.get_domain('abst')
        df.loc[q, p] = df.loc[q, p] - 1
    cost_matrix = df.values.astype(float)  # convert to float array
    
    reg = np.abs(np.min(cost_matrix))
    cost_matrix += reg
    cost_matrix +=  np.ones_like(cost_matrix)
    
    return cost_matrix
 
def generate_hamming_cost_matrix(pairs):
    lst_base = pairs[0].base_labels
    lst_abst = pairs[0].abst_labels
    cost_matrix = np.array([[hamming_distance(x, y)/1. for x in lst_base] for y in lst_abst])
    cost_matrix +=  np.ones_like(cost_matrix)
    return cost_matrix


def costs_visulization(I_relevant, omega, M_base, M_abst):

    pairs = ut.create_pairs(1000, I_relevant, omega, M_base, M_abst)

    cost1 = generate_euclidean_cost_matrix([pairs[0].base_distribution, pairs[0].abst_distribution])
    cost2 = generate_sqeuclidean_cost_matrix([pairs[0].base_distribution, pairs[0].abst_distribution])
    cost3 = generate_quadratic2_cost_matrix([pairs[0].base_distribution, pairs[0].abst_distribution])
    cost4 = generate_omega_cost_matrix(pairs)
    cost5 = generate_hamming_cost_matrix(pairs)

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    axs[0].matshow(cost1)
    axs[0].set_title('Euclidean: |x-y|', fontsize=9)
    axs[1].matshow(cost2)
    axs[1].set_title('Square Euclidean: $|x-y|^2$', fontsize=9)
    axs[2].matshow(cost3)
    axs[2].set_title('$Quadratic: 1/2|x-y|^2$', fontsize=9)

    plt.show() 

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].matshow(cost4)
    axs[0].set_title('Orderd $\omega$-informed', fontsize=9)
    axs[1].matshow(cost5)
    axs[1].set_title('Orderd Hamming', fontsize=9)

    plt.show()
    
    return