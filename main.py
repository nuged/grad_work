from updaters import *
from network import createNetwork, train, test
import yaml
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

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

model = BaseModel(4, epsilon=0.2, seed=90)

for i in range(100):
    if i % 10 == 0:
        startState, actions = model.BestSequence(SEQ_SIZE)
        print '--------------------------------EVAL------------------------------'
    else:
        startState, actions = model.RandomStartSequence(SEQ_SIZE)
        print '-------------------------------TRAIN------------------------------'

    actions = map(int, actions)
    startState = map(int, startState)
    print '\nstart, actions = ', startState, actions
    updater = specificUpdater(STEP, actions, startState)
    baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

    net = createNetwork(baseParameters)
    start = time()
    train(net, 'mnist')
    acc, mse = test(net, 'mnist')
    t = time() - start
    print 'took %f sec, acc=%4.2f, mse=%5.2f\n' % (t, acc, mse)

    with open('logs/mcc_5.txt', 'a') as f:
        if i % 10 == 0:
            print >> f, startState, actions, acc, mse, 'eval'
        else:
            print >> f, startState, actions, acc, mse

    try:
        if i % 10 == 0 and i > 0:
            model.save('models/epsilon-greedy/2e-1')
        else:
            model.update(acc)
    except:
        gc.collect()
        print i, 'failed'