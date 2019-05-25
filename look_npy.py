import numpy as np

FILE = 'stateAction_04'
FILENAME = 'models/category/%s.npy' % FILE

arr = np.load(FILENAME)

print arr