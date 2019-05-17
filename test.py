from updaters import *
from network import createNetwork, train, test
import yaml, utils
import pandas as pd, numpy as np
from time import time, sleep
import utils
from math import ceil
import gc, sys, copy
from reinforcement import BaseModel

IMG_SIZE = 28
WINDOW_SIZE = 10
STEP = 7
SEQ_SIZE = 6
actions = [0, 2, 2, 1, 1, 3]
startState = [14, 14]

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

updater = specificUpdater(STEP, actions, startState)
baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

params = pd.read_csv('test_params.csv', float_precision='high')
first = True
for row in params.iterrows():

    for key, value in row[1].iteritems():
        if params[key].dtype == 'float64':
            baseParameters['TM'][key] = float(value)
        else:
            baseParameters['TM'][key] = int(value)

    baseParameters['SP2']['inputWidth'] = baseParameters['TM']['cellsPerColumn'] * baseParameters['TM']['columnCount']

    net = createNetwork(baseParameters)

    start = time()
    train(net, 'mnist')
    acc, mse = test(net, 'mnist')
    t = time() - start

    print 'took %f sec, acc=%4.2f\n' % (t, acc)

    t = int(t)
    acc = round(acc, 2)

    baseParameters['TM']['time'] = t
    baseParameters['TM']['result'] = acc
    utils.writeResults(baseParameters['TM'], 'test.csv')
    first = False
