import itertools
import joblib

import networkx as nx
import numpy as np
import pandas as pd
import cvxpy as cp
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from pgmpy import inference

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

import scipy.optimize as optimize
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.stats import wasserstein_distance
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

import optimization_functions as optim


def run_cota_grid(all_chains, c, met, kk, ll, mm, exp):
    
    repairs = []
    for path in all_chains:
        for node in path:
            repairs.append(node.data)
    repairs = list(set(repairs))

    if   c == 'Euclidean':
        costM = cts.generate_euclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                    all_chains[0][0].data.abst_distribution])
    elif c == 'Sqeuclidean':
        costM = cts.generate_sqeuclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                      all_chains[0][0].data.abst_distribution])
    elif c == 'Quadratic2':
        costM = cts.generate_quadratic2_cost_matrix([all_chains[0][0].data.base_distribution,
                                                     all_chains[0][0].data.abst_distribution])
    elif c == 'Omega':
        costM = cts.generate_omega_cost_matrix(repairs) 
        
    elif c == 'Hamming':
        costM = cts.generate_hamming_cost_matrix(repairs)
     
    
    costM = costM/np.sum(costM)

    m = costM.shape[0] # dimensions of the abstracted model
    n = costM.shape[1] # dimensions of the base model
    
    obj = cp.Constant(0) 
    processed_pairs, processed_dists, constraints, omega_plans = [], [], [], []
    
    for chain in all_chains:
        
        if exp == 'synth1T' or exp == 'synth1Tinv':
            constraints, partial_obj, processed_pairs, processed_dists = optim.causal_joint_ot_grid_parents(cost_matrix = costM,
                                                                                            chain = chain,
                                                                                            kk = kk,
                                                                                            ll = ll,
                                                                                            mm = mm,
                                                                                            metric = met,
                                                                                            processed_pairs=processed_pairs,
                                                                                            processed_dists=processed_dists,
                                                                                            constraints = constraints,
                                                                                            exp = exp)
        else:
            constraints, partial_obj, processed_pairs, processed_dists = optim.causal_joint_ot_grid(cost_matrix = costM,
                                                                                                chain = chain,
                                                                                                kk = kk,
                                                                                                ll = ll,
                                                                                                mm = mm,
                                                                                                metric = met,
                                                                                                processed_pairs=processed_pairs,
                                                                                                processed_dists=processed_dists,
                                                                                                constraints = constraints)



        obj += partial_obj
        
    prob   = cp.Problem(cp.Minimize(obj), constraints)
    
    result = prob.solve()
    
    all_plans = []
    for variable in prob.variables():
        for val in variable.value.reshape((-1,n,m)):
            all_plans.append(val.T)
    
    all_plans = [np.around(plan, decimals=2) for plan in all_plans if np.isclose(np.sum(plan), 1)]

    return all_plans, result, costM


def run_cota_grid_multi(all_chains, c, met, kk, ll1, ll2, mm, exp):
    
    repairs = []
    for path in all_chains:
        for node in path:
            repairs.append(node.data)
    repairs = list(set(repairs))

    if   c == 'Euclidean':
        costM = cts.generate_euclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                    all_chains[0][0].data.abst_distribution])
    elif c == 'Sqeuclidean':
        costM = cts.generate_sqeuclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                      all_chains[0][0].data.abst_distribution])
    elif c == 'Quadratic2':
        costM = cts.generate_quadratic2_cost_matrix([all_chains[0][0].data.base_distribution,
                                                     all_chains[0][0].data.abst_distribution])
    elif c == 'Omega':
        costM = cts.generate_omega_cost_matrix(repairs) 
        
    elif c == 'Hamming':
        costM = cts.generate_hamming_cost_matrix(repairs)
     
    
    costM = costM/np.sum(costM)

    m = costM.shape[0] # dimensions of the abstracted model
    n = costM.shape[1] # dimensions of the base model
    
    obj = cp.Constant(0) 
    processed_pairs, processed_dists, constraints, omega_plans = [], [], [], []
    
    for chain in all_chains:
        
        if exp == 'synth1T' or exp == 'synth1Tinv':
            constraints, partial_obj, processed_pairs, processed_dists = optim.causal_joint_ot_grid_parents_multi(cost_matrix = costM,
                                                                                            chain = chain,
                                                                                            kk = kk,
                                                                                            ll1 = ll1,
                                                                                            ll2 = ll2,
                                                                                            mm = mm,
                                                                                            metric = met,
                                                                                            processed_pairs=processed_pairs,
                                                                                            processed_dists=processed_dists,
                                                                                            constraints = constraints,
                                                                                            exp = exp)
        else:
            constraints, partial_obj, processed_pairs, processed_dists = optim.causal_joint_ot_grid_multi(cost_matrix = costM,
                                                                                                chain = chain,
                                                                                                kk = kk,
                                                                                                ll1 = ll1,
                                                                                                ll2 = ll2,
                                                                                                mm = mm,
                                                                                                metric = met,
                                                                                                processed_pairs=processed_pairs,
                                                                                                processed_dists=processed_dists,
                                                                                                constraints = constraints)



        obj += partial_obj
        
    prob   = cp.Problem(cp.Minimize(obj), constraints)
    
    result = prob.solve()
    
    all_plans = []
    for variable in prob.variables():
        for val in variable.value.reshape((-1,n,m)):
            all_plans.append(val.T)
    
    all_plans = [np.around(plan, decimals=2) for plan in all_plans if np.isclose(np.sum(plan), 1)]

    return all_plans, result, costM



def run_cota_grid_approx(all_chains, c, met, kk, ll, mm, exp):
    
    repairs = []
    for path in all_chains:
        for node in path:
            repairs.append(node.data)
    repairs = list(set(repairs))

    if   c == 'Euclidean':
        costM = cts.generate_euclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                    all_chains[0][0].data.abst_distribution])
    elif c == 'Sqeuclidean':
        costM = cts.generate_sqeuclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                      all_chains[0][0].data.abst_distribution])
    elif c == 'Quadratic2':
        costM = cts.generate_quadratic2_cost_matrix([all_chains[0][0].data.base_distribution,
                                                     all_chains[0][0].data.abst_distribution])
    elif c == 'Omega':
        costM = cts.generate_omega_cost_matrix(repairs) 
        
    elif c == 'Hamming':
        costM = cts.generate_hamming_cost_matrix(repairs)
     
    
    costM = costM/np.sum(costM)

    m = costM.shape[0] # dimensions of the abstracted model
    n = costM.shape[1] # dimensions of the base model
    
    obj = cp.Constant(0) 
    processed_pairs, processed_dists, constraints, omega_plans = [], [], [], []
    
    for chain in all_chains:
        
        if exp == 'synth1T' or exp == 'synth1Tinv':
            constraints, partial_obj, processed_pairs, processed_dists = optim.causal_joint_ot_grid_parents_approx(cost_matrix = costM,
                                                                                            chain = chain,
                                                                                            kk = kk,
                                                                                            ll = ll,
                                                                                            mm = mm,
                                                                                            metric = met,
                                                                                            processed_pairs=processed_pairs,
                                                                                            processed_dists=processed_dists,
                                                                                            constraints = constraints,
                                                                                            exp = exp)
        else:
            constraints, partial_obj, processed_pairs, processed_dists = optim.causal_joint_ot_grid_approx(cost_matrix = costM,
                                                                                                chain = chain,
                                                                                                kk = kk,
                                                                                                ll = ll,
                                                                                                mm = mm,
                                                                                                metric = met,
                                                                                                processed_pairs=processed_pairs,
                                                                                                processed_dists=processed_dists,
                                                                                                constraints = constraints)



        obj += partial_obj
        
    prob   = cp.Problem(cp.Minimize(obj), constraints)
    
    result = prob.solve()
    
    all_plans = []
    for variable in prob.variables():
        for val in variable.value.reshape((-1,n,m)):
            all_plans.append(val.T)
    
    all_plans = [np.around(plan, decimals=2) for plan in all_plans if np.isclose(np.sum(plan), 1)]

    return all_plans, result, costM

def run_experiments_baselines(ps, c, mode, method, exp):
    
    if   c == 'Euclidean':
        costM = cts.generate_euclidean_cost_matrix([ps[0].base_distribution, ps[0].abst_distribution])         
    elif c == 'Sqeuclidean':
        costM = cts.generate_sqeuclidean_cost_matrix([ps[0].base_distribution, ps[0].abst_distribution])
    elif c == 'Quadratic2':
        costM = cts.generate_quadratic2_cost_matrix([ps[0].base_distribution, ps[0].abst_distribution])
    elif c == 'Omega':
        costM = cts.generate_omega_cost_matrix(ps)
    elif c == 'Hamming':
        costM = cts.generate_hamming_cost_matrix(ps)
    
    
    if mode == 'pairwise' or mode == 'aggregated':
        total_plans = []
        total_cost  = 0
        for pair in ps:
            p, c = optim.pairwise_ot(costM, pair)
            if exp == 'little_lucas':
                p = np.around(p, decimals=2)

            total_plans.append(p)
            total_cost += c

        return total_plans, total_cost, costM
    
    if mode == 'barycentric':
        source, target = [], []
        for pair in ps:
            source.append(pair.base_distribution)
            target.append(pair.abst_distribution)
        source = np.array(source)
        target = np.array(target)

        bary_plan, total_cost = optim.barycentric_ot(costM, source, target, method)
        #bary_plan = np.around(bary_plan, decimals=2)

        
        return bary_plan, total_cost, costM
    
##################################################################################################################################

def run_experiments_cota(all_chains, c, met, lmbd):
    
    repairs = []
    for path in all_chains:
        for node in path:
            repairs.append(node.data)
    repairs = list(set(repairs))

    if   c == 'Euclidean':
        costM = cts.generate_euclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                    all_chains[0][0].data.abst_distribution])
    elif c == 'Sqeuclidean':
        costM = cts.generate_sqeuclidean_cost_matrix([all_chains[0][0].data.base_distribution,
                                                      all_chains[0][0].data.abst_distribution])
    elif c == 'Quadratic2':
        costM = cts.generate_quadratic2_cost_matrix([all_chains[0][0].data.base_distribution,
                                                     all_chains[0][0].data.abst_distribution])
    elif c == 'Omega':
        costM = cts.generate_omega_cost_matrix(repairs) 
    elif c == 'Hamming':
        costM = cts.generate_hamming_cost_matrix(repairs)
     
    
    costM = costM/np.sum(costM)

    m = costM.shape[0] # dimensions of the abstracted model
    n = costM.shape[1] # dimensions of the base model
    
    obj = cp.Constant(0) 
    processed_pairs, processed_dists, constraints, omega_plans = [], [], [], []
 
    for chain in all_chains:
        constraints, partial_obj, processed_pairs, processed_dists = optim.causal_joint_ot(cost_matrix = costM,
                                                                                            chain = chain,
                                                                                            lbda = lmbd, 
                                                                                            metric = met,
                                                                                            processed_pairs=processed_pairs,
                                                                                            processed_dists=processed_dists,
                                                                                            constraints = constraints)





        obj += partial_obj
        
    prob   = cp.Problem(cp.Minimize(obj), constraints)

    try:
        #print('Solver: ECOS')
        result = prob.solve()
    except SolverError:
        print('Solver: SCS')
        result = prob.solve(solver=cp.SCS)
    #result = prob.solve(verbose=False) 

    all_plans = []
    for variable in prob.variables():
        for val in variable.value.reshape((-1,n,m)):
            all_plans.append(val.T)
        
    all_plans = [np.around(plan, decimals=2) for plan in all_plans if np.isclose(np.sum(plan), 1)]
    
    return all_plans, result, costM