import numpy as np
import random
import distances as dists
from IPython.utils import io


def maximum_map_old(plan, labels, form):
    source = [''.join(map(str, tpl)) for tpl in labels[0]]
    target = [''.join(map(str, tpl)) for tpl in labels[1]]

    mapping = {}
    binary_matrix = [[0] * len(source) for _ in range(len(target))]
    
    for j, column in enumerate(source):
        max_value = 0
        max_row = None
        for i, row in enumerate(target):
            if plan[i][j] > max_value:
                max_value = plan[i][j]
                max_row = row
        mapping[column] = max_row
        
        if max_row is not None:
            binary_matrix[target.index(max_row)][j] = 1
    
    if form == 'matrix':
        return binary_matrix
    return mapping


def maximum_map(plan, labels, form):
    source = [''.join(map(str, tpl)) for tpl in labels[0]]
    target = [''.join(map(str, tpl)) for tpl in labels[1]]

    mapping = {}
    binary_matrix = [[0] * len(source) for _ in range(len(target))]
    
    for j, column in enumerate(source):
        max_value = 0
        max_row = None
        zero_columns = []

        for i, row in enumerate(target):
            if plan[i][j] > max_value:
                max_value = plan[i][j]
                max_row = row
            if plan[i][j] == 0:
                zero_columns.append(row)

        if zero_columns and max_row is None:
            max_row = random.choice(zero_columns)
        
        mapping[column] = max_row
        
        if max_row is not None:
            binary_matrix[target.index(max_row)][j] = 1
    
    if form == 'matrix':
        return binary_matrix
    return mapping


def stochastic_map(plan, labels, form):
    
    source = labels[0] #[''.join(map(str, tpl)) for tpl in labels[0]]
    target = labels[1] #[''.join(map(str, tpl)) for tpl in labels[1]]
    
    mapping = {}

    for i, column_label in enumerate(source):
        column = plan[:, i]
        normalized_column = column / (np.sum(column)+0.001)
        mapping[column_label] = list(normalized_column)

    if form == 'matrix':
        return plan
    return mapping

def barycentric_map(plan, labels, form):
    source = labels[0]
    target = labels[1]

    mapping = {}
    binary_matrix = np.zeros((len(target), len(source)), dtype=int)

    for j, column in enumerate(source):
        min_value = float('inf')
        min_row = None
        for i, row in enumerate(target):
            dists = [dists.hamming_distance(row, target[k]) for k in range(len(target))]
            s = np.sum(plan[:, j] * dists[i])
            if s < min_value:
                min_value = s
                min_row = row
        mapping[column] = row

        # Set the corresponding element in the binary matrix to 1
        if min_row is not None:
            binary_matrix[target.index(min_row), j] = 1
            
    mapping = {''.join(map(str, key)): ''.join(map(str, value)) for key, value in mapping.items()}
    
    if form == 'matrix':
        return binary_matrix
    return mapping

def find_mapping(plan, method, order, form):
    
    if method == 'maximum':
        abstraction = maximum_map(plan, order, form)
        
    elif method == 'stochastic':
        abstraction = stochastic_map(plan, order, form)
        
    elif method == 'barycentric':
        abstraction = barycentric_map(plan, order, form)
        
    return abstraction