from updaters import *
from network import createNetwork, train, test
import yaml
import pandas as pd, numpy as np
from time import time, sleep

from math import ceil
import gc, sys, copy
from reinforcement import ESModel

IMG_SIZE = 28
WINDOW_SIZE = 10
STEP = 7
SEQ_SIZE = 8

with open('parameters.yaml', "r") as f, open('logs.txt', 'w') as g:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

model = ESModel(4, seed=90)

for i in range(250):
    startState, actions = model.createESSequence(SEQ_SIZE)
    actions = map(int, actions)
    print '\nstart, actions = ', startState, actions
    updater = specificUpdater(STEP, actions, startState)
    baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

    net = createNetwork(baseParameters)
    start = time()
    train(net, 'mnist')
    acc, mse = test(net, 'mnist')
    t = time() - start
    print 'took %f sec, acc=%4.2f, mse=%5.2f' % (t, acc, mse)
    with open('logs/mcc_2.txt', 'a') as f:
        print >> f, startState, actions, acc, mse

    try:
        model.update(acc)
        if i % 10 == 0:
            model.save('models')
    except:
        gc.collect()