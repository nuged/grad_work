from updaters import *
from network import createNetwork, train, test
import yaml
import pandas as pd
from time import time
from math import ceil
import gc, sys

IMG_SIZE = 28
WINDOW_SIZE = 10
STEP = 7
ACTIONS = ['down', 'down', 'right', 'up', 'up', 'right']

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

updater = specificUpdater(STEP, ACTIONS)

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE
baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

net = createNetwork(baseParameters)
start = time()
train(net, 'mnist')
pctCorrect = test(net, 'mnist')
t = time() - start

print pctCorrect, '\ttook %f sec' % (t)
