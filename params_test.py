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

REGION = 'SP'
NEXT_REGION = 'TM'

params = {'numActiveColumnsPerInhArea' : [40, 60, 80, 100, 120],
          'columnCount' : [1024, 2048, 4096]}

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

updater = specificUpdater(STEP, actions, startState)
baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

first = True
for regionParameters in utils.genParams(baseParameters[REGION], params):
    baseParameters[NEXT_REGION]['inputWidth'] = regionParameters['columnCount']
    baseParameters[NEXT_REGION]['columnCount'] = regionParameters['columnCount']

    net = createNetwork(baseParameters)
    start = time()
    #n_patterns = train(net, 'mnist')
    acc = test(net, 'mnist')
    t = time() - start

    print 'took %f sec, acc=%4.2f\n' % (t, acc)
    break
    utils.writeResults('%s_result.csv' % REGION, regionParameters, n_patterns, acc, t, first)
    first = False
