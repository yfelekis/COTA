import itertools
import joblib

import os
import math
import networkx as nx
import numpy as np
import pandas as pd
import cvxpy as cp
from tqdm import tqdm 

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from sklearn.metrics import mutual_info_score

from pgmpy import inference
from scipy.stats import entropy
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
import plotting
import runs

import optimization_functions as optim


metrics   = ['fro', 'jsd'] 
costs     = ['Omega', 'Hamming']
methods   = ['stochastic']

def results_grid(args, experiment):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll         = args[3]
    mm         = args[4]
    dist_func  = args[5]
    cost_func  = args[6]
    
    combination = str(kk)+'-'+str(ll)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_grid_plans(all_chains, kk, ll, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_grid_maps(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed

def get_cota_grid_plans(chains, kk, ll, mm, dist_func, cost_func, exp):

    visualize = False

    cota_grid_results = []

    for n in range(len(chains)):
        
        p, c, m = runs.run_cota_grid(chains[n], cost_func, dist_func, kk, ll, mm, exp)
       
        avg_plan = np.around(np.mean(p, axis=0), decimals=2)
        cota_grid_results.append(avg_plan)
        
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]

    return cota_grid_results, labels, 0


def get_cota_grid_maps(cota_plans, labels, form = None):
   
    cota_map_results = []

    for i in range(len(cota_plans)):
        
        abst = maps.find_mapping(cota_plans[i], 'stochastic', labels, form)
        cota_map_results.append(abst)

    return cota_map_results

def get_cota_plans_aggregated(chains, kk, ll, mm, dist_func, cost_func, exp):
   
    visualize = False

    cota_grid_results = []
    for n in range(len(chains)):

        p, c, m = runs.run_cota_grid(chains[n], cost_func, dist_func, kk, ll, mm, exp)
        
        cota_grid_results.append(p)
        
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]

    return cota_grid_results, labels, 0

def get_cota_maps_aggregated(cota_plans_agg, labels, form = None):

    aggrageted_map_results = []

    all_absts = []
    for single_plan in cota_plans_agg[0]:
        abst = maps.find_mapping(single_plan, 'stochastic', labels, form)
        all_absts.append(abst)

    average_map = {} 
    for key in all_absts[0]:
        values = [abst[key] for abst in all_absts]
        average_value = [sum(elements) / len(elements) for elements in zip(*values)]
        average_map[key] = average_value
        
    return average_map


#############################################################################################################################

def results_grid_looo(args, experiment, dropped_pair):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll         = args[3]
    mm         = args[4]
    dist_func  = args[5]
    cost_func  = args[6]
    
    combination = str(kk)+'-'+str(ll)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_grid_plans(all_chains, kk, ll, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_grid_maps(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}/{dropped_pair}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed


def results_grid_looo_aggregated(args, experiment, dropped_pair):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll         = args[3]
    mm         = args[4]
    dist_func  = args[5]
    cost_func  = args[6]
    
    combination = str(kk)+'-'+str(ll)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_plans_aggregated(all_chains, kk, ll, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_maps_aggregated(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}/{dropped_pair}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid_agg.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid_agg.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed

################################################################################################################################

def get_plans_pairwise(pairs, experiment):
    
    pairs = [pairs] #?
    pairwise_optimization_results = []
    for n in range(len(pairs)):

        all_avgs  = {}
        AveragePlan, wAveragePlan = {}, {}
        for cost_func in costs:

            p, c, m = runs.run_experiments_baselines(ps = pairs[n],
                                                     c = cost_func,
                                                     mode = 'pairwise',
                                                     method = None,
                                                     exp = experiment)
            
            AveragePlan[cost_func] = np.around(np.mean(p, axis=0), decimals=2)

            sw, tot = 0, 0
            for k in range(len(p)):
                tot += p[k]*np.count_nonzero(p[k])
                sw  += np.count_nonzero(p[k])

            wAveragePlan[cost_func] = np.around(tot/np.sum(sw), decimals=2)

        all_avgs['avg']   = AveragePlan
        all_avgs['wavg']  = wAveragePlan

        pairwise_optimization_results.append(all_avgs)
        
    base_labels, abst_labels = pairs[0][0].base_labels, pairs[0][0].abst_labels
    labels = [base_labels, abst_labels]
    
    return pairwise_optimization_results, labels

def get_maps_pairwise(pwise_plans, labels, form = None):
    
    pairwise_map_results = []
    for i in range(len(pwise_plans)):

        map_results = {}
        for map_method in methods:

            avg_map = {}
            for averaging, logs in pwise_plans[i].items():

                abstraction = {}
                for cost_func, average_plan in logs.items():

                    abst = maps.find_mapping(average_plan, map_method, labels, form)
                    abstraction[cost_func] = abst

                avg_map[averaging] = abstraction

            map_results[map_method] = avg_map

        pairwise_map_results.append(map_results)

    return pairwise_map_results


def get_plans_barycentric(pairs, experiment):
    pairs = [pairs] #?
    barycentric_method = 'l2' #'l2' #'emd', 'lp_emd' 

    barycentric_optimization_results = []
    for n in range(len(pairs)):

        barycentricPlans     = {}
        for cost_func in costs:

            p, c, m = runs.run_experiments_baselines(ps = pairs[n], 
                                                     c = cost_func, 
                                                     mode = 'barycentric', 
                                                     method = barycentric_method,
                                                     exp = experiment)
            barycentricPlans[cost_func] = np.around(p, decimals=2)

        barycentric_optimization_results.append(barycentricPlans)
        
    base_labels, abst_labels = pairs[0][0].base_labels, pairs[0][0].abst_labels
    labels = [base_labels, abst_labels]
    
    return barycentric_optimization_results, labels


def get_maps_barycentric(bary_plans, labels, form = None):
    
    barycentric_map_results = []
    for i in range(len(bary_plans)):

        bmap_results ={}
        for map_method in methods:

            abstraction = {}
            for cost_f, bary_plan in bary_plans[i].items():

                abst = maps.find_mapping(bary_plan, map_method, labels, form)
                abstraction[cost_f] = abst

            bmap_results[map_method] = abstraction
        barycentric_map_results.append(bmap_results)
    
    return barycentric_map_results


def get_maps_aggregated(pairs, experiment, form = None):
    
    pairs = [pairs] #?
    base_labels, abst_labels = pairs[0][0].base_labels, pairs[0][0].abst_labels
    labels = [base_labels, abst_labels]
    
    aggrageted_map_results = []
    for n in range(len(pairs)):

        amap_results     = {}
        for map_method in methods:

            cdict = {}
            for cost_func in costs:

                p, c, m = runs.run_experiments_baselines(ps = pairs[n],
                                                         c = cost_func,
                                                         mode = 'pairwise',
                                                         method = None,
                                                         exp = experiment)

                all_absts = []
                for single_plan in p:
                    abst = maps.find_mapping(single_plan, map_method, labels, form)
                    all_absts.append(abst)

                majority_votes, average_values = {}, {}  
                for key in all_absts[0]:

                    values = [abst[key] for abst in all_absts]
                    if map_method == 'stochastic':
                        average_value = [sum(elements) / len(elements) for elements in zip(*values)]
                        average_values[key] = average_value
                    else:
                        counts = Counter(values)
                        most_common_value, count = counts.most_common(1)[0]
                        majority_votes[key] = most_common_value

                if map_method == 'stochastic':
                    cdict[cost_func] = average_values
                else:
                    cdict[cost_func] = majority_votes

                amap_results[map_method] = cdict
        aggrageted_map_results.append(amap_results)

    return aggrageted_map_results

def get_plans_cota(chains, lmbdas):

    visualize = False

    cota_optimization_results = []
    for n in range(len(chains)):

        lmbda_optimization_results = {}
        for lmbda in lmbdas:
            optimization_results = {}
            for dist_func in metrics:
                all_avgs     = {}
                all_tots     = {}
                w_avg_plan   = {}
                avg_plan     = {}
                for cost_func in costs:
                    
                    p, c, m = runs.run_experiments_cota(chains[n], cost_func, dist_func, lmbda)
                    
                    if visualize == True:
                        plotting.visualize_res(plans = p, cost = c, 
                                      cost_func = cost_func, 
                                      show_values = False, 
                                      base_labels = chains[0][0].base_labels,
                                      abst_labels = chains[0][0].abst_labels)

                    avg_plan[cost_func] = np.around(np.mean(p, axis=0), decimals=2)
                    tot, sw = 0, 0
                    for k in range(len(p)):

                        tot += p[k]*np.count_nonzero(p[k])
                        sw  += np.count_nonzero(p[k])

                    w_avg_plan[cost_func] = np.around(tot/np.sum(sw), decimals=2)
                    

                all_avgs['avg']  = avg_plan
                all_avgs['wavg'] = w_avg_plan

                optimization_results[dist_func] = all_avgs  

            lmbda_optimization_results[lmbda] = optimization_results

        cota_optimization_results.append(lmbda_optimization_results)
    
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]
    
    return cota_optimization_results, labels


def get_maps_cota(cota_plans, labels, form = None):
   
    lmbdas     = cota_plans[0].keys()
    cota_map_results = []

    for i in range(len(cota_plans)):

        lmbda_results = {}
        for lmbda in lmbdas:

            map_results = {}
            for metric, logs in cota_plans[i][lmbda].items():

                avg_method ={}
                for map_method in methods:

                    avg_map = {}
                    for averaging in list(logs.keys()):

                        abstraction = {}
                        for cost_func, average_plan in logs[averaging].items():

                            abst = maps.find_mapping(average_plan, map_method, labels, form)
                            abstraction[cost_func] = abst

                        avg_map[averaging] = abstraction

                    avg_method[map_method] = avg_map

                map_results[metric] = avg_method

            lmbda_results[lmbda] = map_results

        cota_map_results.append(lmbda_results)

    return cota_map_results

################################################################################################################################

def results(mode, args, experiment, dropped_pair):
    
    all_pairs  = args[0]
    all_chains = args[1]
    lmbdas     = args[2]
    
    if mode == 'cota':
        
        cota_plans,ls   = get_plans_cota(all_chains, lmbdas)
        cota_maps       = get_maps_cota(cota_plans, ls)

        cota_plans_dir = f'results/{experiment}/{dropped_pair}'

        os.makedirs(cota_plans_dir, exist_ok=True)

        cota_plans_path = f'{cota_plans_dir}/cota_plans.pkl'
        cota_maps_path = f'{cota_plans_dir}/cota_maps.pkl'

        joblib.dump(cota_plans, cota_plans_path)
        joblib.dump(cota_maps, cota_maps_path)
        
    elif mode == 'pwise':

        pwise_plans,ls = get_plans_pairwise(all_pairs)
        pwise_maps     = get_maps_pairwise(pwise_plans, ls)

        pwise_plans_dir = f'results/{experiment}/{dropped_pair}'
        os.makedirs(pwise_plans_dir, exist_ok=True)

        pwise_plans_path = f'{pwise_plans_dir}/pwise_plans.pkl'
        pwise_maps_path = f'{pwise_plans_dir}/pwise_maps.pkl'
        
        joblib.dump(pwise_plans, pwise_plans_path)
        joblib.dump(pwise_maps, pwise_maps_path)
        
    elif mode == 'bary':
        
        bary_plans,ls  = get_plans_barycentric(all_pairs, experiment)
        bary_maps      = get_maps_barycentric(bary_plans, ls)

        bary_plans_dir = f'results/{experiment}/{dropped_pair}'
        os.makedirs(bary_plans_dir, exist_ok=True)

        bary_plans_path = f'{bary_plans_dir}/bary_plans.pkl'
        bary_maps_path = f'{bary_plans_dir}/bary_maps.pkl'
        
        joblib.dump(bary_plans, bary_plans_path)
        joblib.dump(bary_maps, bary_maps_path)
        
    elif mode == 'agg':
        
        agg_plans,ls = get_plans_pairwise(all_pairs, experiment)
        agg_maps     = get_maps_aggregated(all_pairs, experiment)
       
        agg_plans_dir = f'results/{experiment}/{dropped_pair}'
        os.makedirs(agg_plans_dir, exist_ok=True)

        agg_plans_path = f'{agg_plans_dir}/agg_plans.pkl'
        agg_maps_path = f'{agg_plans_dir}/agg_maps.pkl'
        
        joblib.dump(agg_plans, agg_plans_path)
        joblib.dump(agg_maps, agg_maps_path)
        
        
    elif mode == 'all':
        
        cota_plans,ls   = get_plans_cota(all_chains, lmbdas)
        cota_maps       = get_maps_cota(cota_plans, ls)

        cota_plans_dir = f'results/{experiment}/{dropped_pair}'
        os.makedirs(cota_plans_dir, exist_ok=True)

        cota_plans_path = f'{cota_plans_dir}/cota_plans.pkl'
        cota_maps_path = f'{cota_plans_dir}/cota_maps.pkl'

        joblib.dump(cota_plans, cota_plans_path)
        joblib.dump(cota_maps, cota_maps_path)
        
        pwise_plans,ls = get_plans_pairwise(all_pairs)
        pwise_maps     = get_maps_pairwise(pwise_plans, ls)

        pwise_plans_dir = f'results/{experiment}/{dropped_pair}'
        os.makedirs(pwise_plans_dir, exist_ok=True)

        pwise_plans_path = f'{pwise_plans_dir}/pwise_plans.pkl'
        pwise_maps_path = f'{pwise_plans_dir}/pwise_maps.pkl'
        
        joblib.dump(pwise_plans, pwise_plans_path)
        joblib.dump(pwise_maps, pwise_maps_path)
        
        bary_plans,ls  = get_plans_barycentric(all_pairs)
        bary_maps      = get_maps_barycentric(bary_plans, ls)

        bary_plans_dir = f'results/{experiment}/{dropped_pair}'
        os.makedirs(bary_plans_dir, exist_ok=True)

        bary_plans_path = f'{bary_plans_dir}/bary_plans.pkl'
        bary_maps_path = f'{bary_plans_dir}/bary_maps.pkl'
        
        joblib.dump(bary_plans, bary_plans_path)
        joblib.dump(bary_maps, bary_maps_path)
        
        agg_maps    = get_maps_aggregated(all_pairs)
       
        agg_plans_dir = f'results/{experiment}/{dropped_pair}'
        os.makedirs(agg_plans_dir, exist_ok=True)

        agg_plans_path = f'{agg_plans_dir}/agg_plans.pkl'
        agg_maps_path = f'{agg_plans_dir}/agg_maps.pkl'
        
        joblib.dump(pwise_plans, agg_plans_path)
        joblib.dump(agg_maps, agg_maps_path)
    
    return


def run_experiment(experiment, mode): # needs /{dropped_pair}
    
    pair_path  = f'data/{experiment}/pairs.pkl'
    chain_path = f'data/{experiment}/chains.pkl'

    pairs      = joblib.load(pair_path)
    chains     = joblib.load(chain_path)

    lmbdas     = params.lmbdas[experiment]

    args       = [pairs, chains, lmbdas]
    
    results(mode, args, experiment)
    
    return

################################################################################################################################
"""def entropy(matrix):
    # Flatten the matrix to a 1D array
    flattened_matrix = matrix#.flatten()

    # Compute the probability of each unique element in the matrix
    unique_elements, counts = np.unique(flattened_matrix, return_counts=True)
    probabilities = counts / len(flattened_matrix)

    # Compute the entropy
    entropy_value = -np.sum(probabilities * np.log2(probabilities))

    return entropy_value
    
    def get_cota_grid_plans1(chains, kk, ll, mm, dist_func, cost_func):

    visualize = False

    lst_base = chains[0][0][0].data.base_labels
    lst_abst = chains[0][0][0].data.abst_labels
    
    r = len(chains[0][0][0].data.abst_dict)
    c = len(chains[0][0][0].data.base_dict)
    
    cota_grid_results = []
    
    
    for n in range(len(chains)):
        omega_matrix, om_plan_sum = [], []
        for z, path in enumerate(chains[n]):
            for q, node in enumerate(path):
                omega_mat = np.zeros((r, c))
                if q < len(path)-1:
                    s            = node.get_child(path).data.get_domain('base')
                    t            = node.get_child(path).data.get_domain('abst')
                    pos_t, pos_s = ut.find_positions(lst_abst,t), ut.find_positions(lst_base,s)
                    sub_indices  = np.ix_(pos_t, pos_s)
                    
                    omega_mat[sub_indices[0], sub_indices[1]] = 1
                    omega_matrix.append(omega_mat.ravel())

                    mass_base = np.sum([node.data.base_distribution[ind_i] for ind_i in pos_s])
                    mass_abst = np.sum([node.data.abst_distribution[ind_j] for ind_j in pos_t])

                    bound     = min(mass_base, mass_abst)
                    om_plan_sum.append(bound)

                  
        p, c, m = runs.run_cota_grid(chains[n], cost_func, dist_func, kk, ll, mm)
        
        diffs = []
        for i, pl in enumerate(p):
            
            rows, cols = pl.shape
            
            
            pl_tilda = np.multiply(pl.ravel(),omega_matrix[i])/om_plan_sum[i]
            pl_tilda = pl_tilda/np.sum(pl_tilda)
            
            entropy_pl       = entropy(pl.ravel())/math.log2(rows*cols)
            entropy_pl_tilda = entropy(pl_tilda)/math.log2(rows*cols)
            
            pl_tilda[np.isnan(pl_tilda)] = np.nanmean(pl)
            pl[np.isnan(pl)] = np.nanmean(pl)

            diff =  entropy_pl-entropy_pl_tilda
            diff2 = mutual_info_score(pl.ravel(), pl_tilda)

            #if not math.isnan(diff2):
                #diffs.append(diff2)
            if not math.isnan(entropy_pl_tilda):
                diffs.append(entropy_pl_tilda)
                
        avg_plan = np.around(np.mean(p, axis=0), decimals=2)
        
        avg_entropy_diff = np.sum(diffs)/len(diffs)
        #print('avg_entropy_diff',avg_entropy_diff)
        cota_grid_results.append(avg_plan)
        
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]
    
    return cota_grid_results, labels, avg_entropy_diff
"""









##### RUN FOR MULTIPLE LAMBDAS #####
def results_grid_multi(args, experiment):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll1        = args[3]
    ll2        = args[4]
    mm         = args[5]
    dist_func  = args[6]
    cost_func  = args[7]
    
    combination = str(kk)+'-'+str(ll1)+'-'+str(ll2)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_grid_plans_multi(all_chains, kk, ll1, ll2, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_grid_maps(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed

def get_cota_grid_plans_multi(chains, kk, ll1, ll2, mm, dist_func, cost_func, exp):

    visualize = False

    cota_grid_results = []

    for n in range(len(chains)):
        
        p, c, m = runs.run_cota_grid_multi(chains[n], cost_func, dist_func, kk, ll1, ll2, mm, exp)
       
        avg_plan = np.around(np.mean(p, axis=0), decimals=2)
        cota_grid_results.append(avg_plan)
        
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]

    return cota_grid_results, labels, 0


def get_cota_plans_aggregated_multi(chains, kk, ll1, ll2, mm, dist_func, cost_func, exp):
   
    visualize = False

    cota_grid_results = []
    for n in range(len(chains)):

        p, c, m = runs.run_cota_grid_multi(chains[n], cost_func, dist_func, kk, ll1, ll2, mm, exp)
        
        cota_grid_results.append(p)
        
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]

    return cota_grid_results, labels, 0

#############################################################################################################################

def results_grid_looo_multi(args, experiment, dropped_pair):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll1        = args[3]
    ll2        = args[4]
    mm         = args[5]
    dist_func  = args[6]
    cost_func  = args[7]
    
    combination = str(kk)+'-'+str(ll1)+'-'+str(ll2)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_grid_plans_multi(all_chains, kk, ll1, ll2, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_grid_maps(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}/{dropped_pair}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed


def results_grid_looo_aggregated_multi(args, experiment, dropped_pair):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll1        = args[3]
    ll2        = args[4]
    mm         = args[5]
    dist_func  = args[6]
    cost_func  = args[7]
    
    combination = str(kk)+'-'+str(ll1)+'-'+str(ll2)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_plans_aggregated_multi(all_chains, kk, ll1, ll2, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_maps_aggregated(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}/{dropped_pair}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid_agg.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid_agg.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed

################################################################################################################################
                                                 ### APPROXIMATE COTA ###
################################################################################################################################

def results_grid_approx(args, experiment):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll         = args[3]
    mm         = args[4]
    dist_func  = args[5]
    cost_func  = args[6]
    
    combination = str(kk)+'-'+str(ll)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_grid_plans_approx(all_chains, kk, ll, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_grid_maps(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed

def get_cota_grid_plans_approx(chains, kk, ll, mm, dist_func, cost_func, exp):

    visualize = False

    cota_grid_results = []

    for n in range(len(chains)):
        
        p, c, m = runs.run_cota_grid_approx(chains[n], cost_func, dist_func, kk, ll, mm, exp)
       
        avg_plan = np.around(np.mean(p, axis=0), decimals=2)
        cota_grid_results.append(avg_plan)
        
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]

    return cota_grid_results, labels, 0


def get_cota_plans_aggregated_approx(chains, kk, ll, mm, dist_func, cost_func, exp):
   
    visualize = False

    cota_grid_results = []
    for n in range(len(chains)):

        p, c, m = runs.run_cota_grid_approx(chains[n], cost_func, dist_func, kk, ll, mm, exp)
        
        cota_grid_results.append(p)
        
    base_labels, abst_labels = chains[0][0][0].data.base_labels, chains[0][0][0].data.abst_labels
    labels = [base_labels, abst_labels]

    return cota_grid_results, labels, 0


#############################################################################################################################

def results_grid_looo_approx(args, experiment, dropped_pair):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll         = args[3]
    mm         = args[4]
    dist_func  = args[5]
    cost_func  = args[6]
    
    combination = str(kk)+'-'+str(ll)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_grid_plans_approx(all_chains, kk, ll, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_grid_maps(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}/{dropped_pair}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed


def results_grid_looo_aggregated_approx(args, experiment, dropped_pair):
    
    all_pairs  = args[0]
    all_chains = args[1]
    kk         = args[2]
    ll         = args[3]
    mm         = args[4]
    dist_func  = args[5]
    cost_func  = args[6]
    
    combination = str(kk)+'-'+str(ll)+'-'+str(mm)

    grid_plans,ls,aed   = get_cota_plans_aggregated_approx(all_chains, kk, ll, mm, dist_func, cost_func, experiment)
    grid_maps           = get_cota_maps_aggregated(grid_plans, ls)

    grid_cota_plans_dir = f'results/{experiment}/{combination}/{dropped_pair}'
    os.makedirs(grid_cota_plans_dir, exist_ok=True)

    cota_plans_path_grid = f'{grid_cota_plans_dir}/cota_plans_grid_agg.pkl'
    cota_maps_path_grid = f'{grid_cota_plans_dir}/cota_maps_grid_agg.pkl'

    joblib.dump(grid_plans, cota_plans_path_grid)
    joblib.dump(grid_maps, cota_maps_path_grid)
    
    return aed

