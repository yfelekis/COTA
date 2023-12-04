import numpy as np

experiments = {
                1: 'synth1',
                2: 'synth2',
                3: 'synth1inv',
                4: 'synthX1',
                5: 'lucas',
                6: 'little_lucas',
                7: 'battery_discrete',
                8: 'battery_continuous',
                9: 'synth1T',
                10: 'synth1Tinv'
              }

n_simuls = {
            'synth1': 50,
            'synth2': 2,
            'synth1inv': 2,
            'synthX1': 2,
            'lucas' : 1,
            'little_lucas': 20,
            'battery_discrete' : 1,
            'battery_continuous': 1,
            'synth1T':50,
            'synth1Tinv':50
            }

n_samples  = {
            'synth1': 10000,
            'synth2': 10000,
            'synth1inv': 100000,
            'synthX1': 1000,
            'lucas' : 100000,
            'little_lucas': 10000,
            'battery_discrete' : 100000,
            'battery_continuous': 100000,
            'synth1T': 10000,
            'synth1Tinv': 10000
            }

lmbdas    = {
            'synth1': [0.7],#np.linspace(0, 1, 10),
            'synth2': [0.7],#np.linspace(0, 1, 10),
            'synth1inv': [0.7],#np.linspace(0, 1, 10),
            'synthX1': np.linspace(0, 1, 10),
            'lucas' : np.linspace(0, 1, 4),
            'little_lucas': [0.0],#np.linspace(0, 1, 4),
            'battery_discrete': [0.7],#np.linspace(0, 1, 5),
            'battery_continuous': [1.0],#np.linspace(0, 1, 10)
            'synth1T':[0.7],
            'synth1Tinv':[0.7]
            }


Zb0 = [.7, .7, 1., 1., .2, .2, 1., 1.]
Za0 = [.9, 1., .66, 1.]

Zb1 = [1., 1., .3, .3, 1., 1., .8, .8]
Za1 = [1., .1, 1., .34]

M00 = np.array([[0.700106, 0.700106, 0.0, 0.0, 0.200327, 0.200327, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.660141, 0.660141, 0.0, 0.0, 0.200327, 0.200327, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

M11 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.100056, 0.100056, 0.0, 0.0, 0.100056, 0.100056],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.299894, 0.299894, 0.0, 0.0, 0.339859, 0.339859]])