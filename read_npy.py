import numpy as np
import os

stateActions = []
policies = []
counts = []

os.chdir('models/epsilon-greedy/2e-1')
for file in os.listdir('.'):
    print file[-6:]
    if file.startswith('policy'):
        policies.append(np.load(file))
    elif file.startswith('stateActionC'):
        counts.append(np.load(file))
    else:
        stateActions.append(np.load(file))

print stateActions[-1], '\n', policies[-1]