from updaters import *
from network import *
import yaml, utils
from time import time
import gc
from reinforcement import BaseModel, CategoryModel

IMG_SIZE = 28
WINDOW_SIZE = 10
STEP = 7
SEQ_SIZE = 7    # num of positions
actions = [0, 2, 2, 1, 3, 3]
startState = [7, 7]

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

baseParameters['sensor']['explorer'] = yaml.dump(["regions.secondExplorer", {'start' : startState,
                                                                             'length' : SEQ_SIZE,
                                                                             'step' : STEP}])

model = CategoryModel(10, epsilon=0.1, ignoreCategory=True)

net = createNetwork(baseParameters)
sensor = net.regions['sensor']
classifier = net.regions['CLS']
explorer = sensor.getSelf().explorer[2]

trainLosses = []
testLosses = []

for i in range(3):
    acc, first, epochClsLosses = modifiedTrain(net, model, startState, SEQ_SIZE, 'mnist')
    trainLosses.append(epochClsLosses)
    print 'Train acc:\t%3.1f' % acc

    acc, first, pos, testEpochLoss = modifiedTest(net, model, startState, SEQ_SIZE, 'mnist')
    testLosses.append(testEpochLoss)
    print 'Test acc:\t%3.1f' % acc
    print 'by step:\t', first

    for j in range(10):
        actPath = model.createSequence(j, startState, SEQ_SIZE, random=False, store=True)
        path = []
        for state, act in model.stateActionSequence[::-1]:
            path.append(state)
        print 'path:\t', path
        break
        # print '\tres:\t', pos.mean(axis=2)[j][:-1]
    print '\n'

with open('test_losses.txt', 'w') as f:
    print >> f, testLosses

with open('train_losses.txt', 'w') as f:
    for epoch in trainLosses:
        for item in epoch:
            f.write('%f,' % item)
        f.write('\n')

exit(0)

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