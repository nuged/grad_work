from updaters import *
from network import createNetwork, train, test
import yaml
from time import time
import gc
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

size = 0
model = BaseModel(4, epsilon=0.4, seed=333, historySize=size)

for i in range(180):
    if i == 60:
        size = 1
        model = BaseModel(4, epsilon=0.4, seed=333, historySize=size)

    if i == 120:
        size = 2
        model = BaseModel(4, epsilon=0.4, seed=333, historySize=size)

    if i % 5 == 0:
        startState, actions = model.BestSequence(SEQ_SIZE)
        print '--------------------------------EVAL------------------------------'
    else:
        startState, actions = model.CenterSequence(SEQ_SIZE)
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
    print 'took %f sec, acc=%4.2f\n' % (t, acc)

    with open('logs/historic/size_%d_1.txt' % size, 'a') as f:
        if i % 5 == 0:
            print >> f, startState, actions, acc, 'eval'
        else:
            print >> f, startState, actions, acc, 'train'

    try:
        if i % 5 == 0 and i > 0:
            model.save('models/historic/size_%d' % size)
        model.update(acc)
    except:
        gc.collect()
        print i, 'failed'