import numpy as np
import os

stateActions = []
policies = []

os.chdir('models/epsilon-greedy')
for file in os.listdir('.'):
    if file.startswith('policy'):
        policies.append(np.load(file))
    else:
        stateActions.append(np.load(file))

print stateActions[-1], policies[-1]