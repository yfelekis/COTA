import itertools
import joblib

import networkx as nx
import numpy as np
import pandas as pd
import cvxpy as cp
import math
import random

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from pgmpy import inference

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

import scipy.optimize as optimize
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from scipy.optimize import linprog
from scipy import stats

from collections import Counter
from IPython.utils import io
from cvxpy.error import SolverError
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

import maps
from src.examples import smokingmodels as sm
import modularized_utils as ut
import cost_functions_utils as cts
import classes as cls
import params

        
def regularizer(next_plan, norm_curr_plan, metric):
    
    if metric == 'fro':
        
        return cp.sum(cp.norm(next_plan - norm_curr_plan, 'fro'))

    elif metric == 'jsd':
        
        med  = .5*(norm_curr_plan + next_plan)
        return cp.sum(.5*cp.kl_div(norm_curr_plan, med) + .5*cp.kl_div(next_plan, med)) 
    

def causal_joint_ot_grid(cost_matrix, chain, kk, ll, mm, metric, processed_pairs, processed_dists, constraints):

    m = cost_matrix.shape[0] # dimensions of the abstracted model
    n = cost_matrix.shape[1] # dimensions of the base model
    k = len(chain)

    lst_base, lst_abst = chain[0].data.base_labels, chain[0].data.abst_labels

    '''omega_matrix is the binary matrix defined by omega and 
    om_plan_sum is the normalizing constant'''
    
    omega_matrix, om_plan_sum = [], []
    for z, node in enumerate(chain):
        
        if z < len(chain)-1:
            s            = node.get_child(chain).data.get_domain('base')
            t            = node.get_child(chain).data.get_domain('abst')
            pos_t, pos_s = ut.find_positions(lst_abst,t), ut.find_positions(lst_base,s)
            sub_indices  = np.ix_(pos_t, pos_s)

            omega_mat = np.zeros((m, n))
            omega_mat[sub_indices[0], sub_indices[1]] = 1
            omega_matrix.append(omega_mat.ravel())

            mass_base = np.sum([node.data.base_distribution[k] for k in pos_s])
            mass_abst = np.sum([node.data.abst_distribution[j] for j in pos_t])
           
            bound     = min(mass_base, mass_abst)
            om_plan_sum.append(bound)
        
    C     = cp.Constant(cost_matrix)
        
    kappa = cp.Constant(kk)
    lmbda = cp.Constant(ll)
    mu    = cp.Constant(mm)
    
    plans = cp.Variable(m*n*k, nonneg=True) 
    
    for i, node in enumerate(chain): 
        
        start_idx = i*m*n
        end_idx   = (i+1)*m*n
        plan      = cp.reshape(plans[start_idx:end_idx], (m, n))
        
        if node not in processed_pairs:
            constraints.append(cp.sum(plan, axis = 0) == node.data.base_distribution)
            constraints.append(cp.sum(plan, axis = 1) == node.data.abst_distribution)
            constraints.append(cp.sum(plan) == 1)
            
    obj_list = []
    for i in range(k-1):
        p      = plans[i*m*n:(i+1)*m*n]
        p_norm = cp.multiply(p, omega_matrix[i])/om_plan_sum[i]
        p_next = plans[(i+1)*m*n:(i+2)*m*n]
        
        curr_plan      = cp.reshape(p, (m, n))
        norm_curr_plan = cp.reshape(cp.multiply(p, omega_matrix[i])/om_plan_sum[i], (m,n))
        next_plan      = cp.reshape(p_next, (m, n))
        
        ot_term, reg_term, entropy = cp.Constant(0), cp.Constant(0), cp.Constant(0) 
        if chain[i] not in processed_pairs:
            
            ot_term = cp.sum(cp.multiply(C, curr_plan))
            entropy = cp.sum(cp.entr(curr_plan))/(math.log2(m*n))

            processed_pairs.append(chain[i])

        if (chain[i], chain[i+1]) not in processed_dists:

            reg_term = regularizer(next_plan, norm_curr_plan, metric)
            processed_dists.append((chain[i],chain[i+1]))

        obj_i = kappa*ot_term + lmbda*reg_term - mu*entropy
        
        obj_list.append(obj_i)
    
    partial_obj  = cp.sum(obj_list) 
    
    p_last    = plans[(k-1)*m*n:k*m*n]
    last_plan = cp.reshape(p_last, (m, n))
    
    #ot_term - mu*entropy for the last plan
    partial_obj += kappa*cp.sum(cp.multiply(C,last_plan)) - mu*(cp.sum(cp.entr(last_plan))/(math.log2(m*n)))
    
    return constraints, partial_obj, processed_pairs, processed_dists

def causal_joint_ot_grid_parents(cost_matrix, chain, kk, ll, mm, metric, processed_pairs, processed_dists, constraints, exp):

    m = cost_matrix.shape[0] # dimensions of the abstracted model
    n = cost_matrix.shape[1] # dimensions of the base model
    k = len(chain)
    
    if chain[1].data.iota_base.intervention == {'Tar': 0}:
        mask = 'M00'
        #M = params.M00 
    elif chain[1].data.iota_base.intervention == {'Tar': 1}:
        mask = 'M11'
        #M = params.M11
    
    M = ut.load_mask(experiment = exp, mask = mask, order = 'shuff')
   
    lst_base, lst_abst = chain[0].data.base_labels, chain[0].data.abst_labels
        
    C     = cp.Constant(cost_matrix)
        
    kappa = cp.Constant(kk)
    lmbda = cp.Constant(ll)
    mu    = cp.Constant(mm)
    
    plans = cp.Variable(m*n*k, nonneg=True) 
    
    for i, node in enumerate(chain): 
        
        start_idx = i*m*n
        end_idx   = (i+1)*m*n
        plan      = cp.reshape(plans[start_idx:end_idx], (m, n))
        
        if node not in processed_pairs:
            constraints.append(cp.sum(plan, axis = 0) == node.data.base_distribution)
            constraints.append(cp.sum(plan, axis = 1) == node.data.abst_distribution)
            constraints.append(cp.sum(plan) == 1)
            
    obj_list = []
    for i in range(k-1):
        p      = plans[i*m*n:(i+1)*m*n]
        #p_norm = cp.multiply(p, omega_matrix[i])/om_plan_sum[i]
        p_next = plans[(i+1)*m*n:(i+2)*m*n]
        
        curr_plan      = cp.reshape(p, (m, n))
        norm_curr_plan = cp.multiply(curr_plan, M)
        next_plan      = cp.reshape(p_next, (m, n))
        
        ot_term, reg_term, entropy = cp.Constant(0), cp.Constant(0), cp.Constant(0) 
        if chain[i] not in processed_pairs:
            
            ot_term = cp.sum(cp.multiply(C, curr_plan))
            entropy = cp.sum(cp.entr(curr_plan))/(math.log2(m*n))

            processed_pairs.append(chain[i])

        if (chain[i], chain[i+1]) not in processed_dists:

            reg_term = regularizer(next_plan, norm_curr_plan, metric)
            processed_dists.append((chain[i],chain[i+1]))

        obj_i = kappa*ot_term + lmbda*reg_term - mu*entropy
        
        obj_list.append(obj_i)
    
    partial_obj  = cp.sum(obj_list) 
    
    p_last    = plans[(k-1)*m*n:k*m*n]
    last_plan = cp.reshape(p_last, (m, n))
    
    #ot_term - mu*entropy for the last plan
    partial_obj += kappa*cp.sum(cp.multiply(C,last_plan)) - mu*(cp.sum(cp.entr(last_plan))/(math.log2(m*n)))
    
    return constraints, partial_obj, processed_pairs, processed_dists
        
def pairwise_ot(cost_matrix, pair):
    m = cost_matrix.shape[0]
    n = cost_matrix.shape[1] 

    C       = cp.Parameter(cost_matrix.shape)
    C.value = cost_matrix 

    Plans     = cp.Variable((m,n), nonneg=True) 
    
    constraints = [cp.sum(Plans, axis = 0) == pair.base_distribution,
                   cp.sum(Plans, axis = 1) == pair.abst_distribution,
                   cp.sum(Plans) == 1] 
    
    obj = cp.sum(cp.multiply(C,Plans)) #- .1*cp.sum(cp.entr(Plans))/(math.log2(m*n))
    
    prob = cp.Problem(cp.Minimize(obj), constraints) 
    result = prob.solve()
    
    return Plans.value, result

def bary_computation(source, target, bary_type):
    
    k = 0
    barycenters = []
    for batch in [source,target]:
        
        n = len(batch[0]) # nb bins

        # creating matrix A containing all distributions
        A = np.vstack((batch)).T
        n_distributions = A.shape[1]

        # loss matrix + normalization
        M = ot.utils.dist0(n)
        M /= M.max()

        weights = np.random.random(n_distributions)
        weights /= weights.sum()

        if bary_type == 'l2': # L2 Barycenter (l2)
            bary = A.dot(weights)
            
        elif bary_type == 'emd': # Regular Wasserstein Barycenter (emd)
            reg  = 1e-3
            bary = ot.bregman.barycenter(A, M, reg, weights)
        
        elif bary_type == 'lp_emd': # LP Wasserstein Barycenter (lp_emd)
            bary = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=False)
        
        else:
            raise Exception('Invalid barycenter option')


        barycenters.append(bary)
        
    return barycenters


def barycentric_ot(cost_matrix, source_batch, target_batch, bary_type):
    
    s, t = bary_computation(source_batch, target_batch, bary_type)
    
    m = cost_matrix.shape[0]
    n = cost_matrix.shape[1]
    
    C = cp.Parameter(cost_matrix.shape)
    C.value = cost_matrix
    
    Plans = cp.Variable((m,n), nonneg=True)

    constraints = [cp.sum(Plans, axis = 0) == s, cp.sum(Plans, axis = 1) == t, cp.sum(Plans) == 1] 
    
    obj = cp.sum(cp.multiply(C,Plans)) #- .1*cp.sum(cp.entr(Plans))/(math.log2(m*n))
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    result = prob.solve()
    
    return Plans.value, result

##################################################################################################################################

def causal_joint_ot(cost_matrix, chain, lbda, metric, processed_pairs, processed_dists, constraints):

    m = cost_matrix.shape[0] # dimensions of the abstracted model
    n = cost_matrix.shape[1] # dimensions of the base model
    k = len(chain)

    lst_base, lst_abst = chain[0].data.base_labels, chain[0].data.abst_labels

    '''omega_matrix is the binary matrix defined by omega and 
    om_plan_sum is the normalizing constant'''
    
    omega_matrix, om_plan_sum = [], []
    for z, node in enumerate(chain):
        
        if z < len(chain)-1:
            s            = node.get_child(chain).data.get_domain('base')
            t            = node.get_child(chain).data.get_domain('abst')
            pos_t, pos_s = ut.find_positions(lst_abst,t), ut.find_positions(lst_base,s)
            sub_indices  = np.ix_(pos_t, pos_s)

            omega_mat = np.zeros((m, n))
            omega_mat[sub_indices[0], sub_indices[1]] = 1
            omega_matrix.append(omega_mat.ravel())

            mass_base = np.sum([node.data.base_distribution[k] for k in pos_s])
            mass_abst = np.sum([node.data.abst_distribution[j] for j in pos_t])
           
            bound     = min(mass_base, mass_abst)
            om_plan_sum.append(bound)
        
    C     = cp.Constant(cost_matrix)
    
    kappa = random.uniform(0, 1-lbda)
    mu = 1 - lbda - kappa
    
    kappa = cp.Constant(kappa)
    lmbda = cp.Constant(lbda)
    mu    = cp.Constant(mu)
    
    
    plans = cp.Variable(m*n*k, nonneg=True) 
    
    for i, node in enumerate(chain): 
        
        start_idx = i*m*n
        end_idx   = (i+1)*m*n
        plan      = cp.reshape(plans[start_idx:end_idx], (m, n))
        
        if node not in processed_pairs:
            constraints.append(cp.sum(plan, axis = 0) == node.data.base_distribution)
            constraints.append(cp.sum(plan, axis = 1) == node.data.abst_distribution)
            constraints.append(cp.sum(plan) == 1)
            
    obj_list = []
    for i in range(k-1):
        p      = plans[i*m*n:(i+1)*m*n]
        p_norm = cp.multiply(p, omega_matrix[i])/om_plan_sum[i]
        p_next = plans[(i+1)*m*n:(i+2)*m*n]
        
        curr_plan      = cp.reshape(p, (m, n))
        norm_curr_plan = cp.reshape(cp.multiply(p, omega_matrix[i])/om_plan_sum[i], (m,n))
        next_plan      = cp.reshape(p_next, (m, n))
        
        ot_term, reg_term, entropy = cp.Constant(0), cp.Constant(0), cp.Constant(0) 
        if chain[i] not in processed_pairs:
            
            ot_term = cp.sum(cp.multiply(C, curr_plan))
            entropy = cp.sum(cp.entr(curr_plan))/(math.log2(m*n))

            processed_pairs.append(chain[i])

        if (chain[i], chain[i+1]) not in processed_dists:

            reg_term = regularizer(next_plan, norm_curr_plan, metric)
            processed_dists.append((chain[i],chain[i+1]))

        obj_i = kappa*ot_term + lmbda*reg_term - mu*entropy
        
        obj_list.append(obj_i)
    
    partial_obj  = cp.sum(obj_list) 
    
    p_last    = plans[(k-1)*m*n:k*m*n]
    last_plan = cp.reshape(p_last, (m, n))
    
    #ot_term - mu*entropy for the last plan
    partial_obj += kappa*cp.sum(cp.multiply(C,last_plan)) - mu*(cp.sum(cp.entr(last_plan))/(math.log2(m*n)))
    
    return constraints, partial_obj, processed_pairs, processed_dists