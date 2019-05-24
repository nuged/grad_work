import utils
from updaters import *
from network import createNetwork, train, test
import yaml
from time import time

IMG_SIZE = 28
WINDOW_SIZE = 18
STEP = 12
actions = [1, 3, 1]
startState = [0, 0]

REGION = 'TM'

params = {'initialPerm' : [0.1, 0.5, 0.9],
          'connectedPerm' : [0.1, 0.5, 0.9],
          'permanenceInc' : [0.001, 0.01, 0.1],
          'permanenceDec' : [0.001, 0.01, 0.1]}

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

updater = specificUpdater(STEP, actions, startState)
baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

first = True
for regionParameters in utils.genParams(baseParameters[REGION], params):
    if REGION == 'SP':
        baseParameters['TM']['inputWidth'] = regionParameters['columnCount']
        baseParameters['TM']['columnCount'] = regionParameters['columnCount']
        baseParameters['SP2']['inputWidth'] = regionParameters['columnCount'] * baseParameters['TM']['cellsPerColumn']
    elif REGION == 'TM':
        baseParameters['SP2']['inputWidth'] = regionParameters['columnCount'] * baseParameters['TM']['cellsPerColumn']

    baseParameters['TM']['predictedSegmentDecrement'] = baseParameters['TM']['permanenceInc'] * 0.05

    net = createNetwork(baseParameters)
    start = time()
    n_patterns = train(net, 'mnist')
    acc = test(net, 'mnist')
    t = time() - start

    print 'took %f sec, acc=%4.2f\n' % (t, acc)

    utils.writeResults('%s_result.csv' % REGION, baseParameters['TM'], n_patterns, acc, t, first)

