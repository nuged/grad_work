import pandas as pd
import numpy as np
import yaml, utils
from network import createNetwork, train, test
from time import time
from updaters import *

REGION = 'TM'
N = 4

IMG_SIZE = 28
WINDOW_SIZE = 18
STEP = 12
actions = [1, 3, 1]
startState = [0, 0]

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

updater = specificUpdater(STEP, actions, startState)
baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

params = pd.read_csv('%s_result.csv' % REGION)
results = params.result.copy()
results.sort()
lowb = results.iloc[-N]
params = params[params.result >= lowb]

first = True
for i, row in params.iterrows():
    for key, value in row.iteritems():
        if value.is_integer():
            baseParameters[REGION][key] = int(value)
        else:
            baseParameters[REGION][key] = float(value)

    if REGION == 'SP':
        baseParameters['TM']['inputWidth'] = baseParameters['SP']['columnCount']
        baseParameters['TM']['columnCount'] = baseParameters['SP']['columnCount']
        baseParameters['SP2']['inputWidth'] = baseParameters['TM']['columnCount'] * baseParameters['TM']['cellsPerColumn']
    elif REGION == 'TM':
        baseParameters['SP2']['inputWidth'] = baseParameters['SP']['columnCount'] * baseParameters['TM']['cellsPerColumn']

    baseParameters[REGION].pop('numPatterns')
    baseParameters[REGION].pop('time')
    baseParameters[REGION].pop('result')

    net = createNetwork(baseParameters)
    start = time()
    n_patterns = train(net, 'mnist')
    acc = test(net, 'mnist')
    t = time() - start

    print 'took %f sec, acc=%4.2f\n' % (t, acc)

    utils.writeResults('%s_result_best.csv' % REGION, baseParameters[REGION], n_patterns, acc, t, first)
    first = False

