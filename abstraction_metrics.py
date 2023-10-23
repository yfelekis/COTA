import itertools
import random
import numpy as np
from pgmpy import inference
from IPython.utils import io

def map_rdist(m1, m2):
    distance = 0
    common_keys = set(m1.keys()) & set(m2.keys())
    
    for key in common_keys:
        if m1[key] != m2[key]:
            distance += 1
    
    return distance

def map_wdist(m1, m2, important_keys=None):
    distance = 0
    common_keys = set(m1.keys()) & set(m2.keys())

    for key in common_keys:
        if m1[key] != m2[key]:
            if important_keys and key in important_keys:
                distance += 2  # Increase the distance by 2 for important keys
            else:
                distance += 1

    return distance

def map_ndist(m1, m2, important_keys=None):
    distance = 0
    common_keys = set(m1.keys()) & set(m2.keys())
    num_common_keys = len(common_keys)

    for key in common_keys:
        if m1[key] != m2[key]:
            if important_keys and key in important_keys:
                distance += 2  # Increase the distance by 2 for important keys
            else:
                distance += 1

    # Calculate maximum possible distance
    max_distance = num_common_keys
    if important_keys:
        max_distance += 2 * len(set(important_keys) & common_keys)

    # Normalize the distance
    normalized_distance = distance / (max_distance+0.00000001)

    return normalized_distance

def discrete_pushforward(abstraction, p_base, p_abst_keys, key_transform=None, value_accumulation=None):
    pf = {}
    for key in p_abst_keys:
        pf[key] = 0
    for key, value in p_base.items():
        if key in abstraction:
            if key_transform:
                mapped_key = key_transform(abstraction[key])
            else:
                mapped_key = abstraction[key]
            if value_accumulation:
                pf[mapped_key] = value_accumulation(new_dict[mapped_key], value)
            else:
                pf[mapped_key] += value
    return list(pf.values())

def stochastic_pushforward(abstraction, p_base, p_abst_keys):

    pf = {}
    for key in p_abst_keys:
        pf[key] = 0.0
        for d1_key, d1_values in abstraction.items():
            for i, k in enumerate(p_abst_keys):
                if d1_values[i] > 0:
                    if key == k:
                        pf[key] += d1_values[i] * p_base[d1_key]
    
    
    pushf = list(pf.values())
    
    if all(element == 0 for element in pushf):
        pushf_epsilon = []
        for el in pushf:
            pushf_epsilon.append(el + 0.00001)
        return pushf_epsilon #push_f
    
    else:
        return pushf

    #return pushf

def to_tuples(dic, mod):
    
    binary_dict = {}

    for key, value in dic.items():
        #binary_key = tuple(int(bit) for bit in key)
        binary_key = tuple(bit for bit in key) #for continuous
        if mod == 'stochastic':
            binary_value = value
        else:
            binary_value = tuple(int(bit) for bit in value)
            
        binary_dict[binary_key] = binary_value
    
    return binary_dict