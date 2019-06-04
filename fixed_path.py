from network import *
import yaml
from time import time

IMG_SIZE = 28
WINDOW_SIZE = 10
STEP = 7
SEQ_SIZE = 7    # num of positions
actions = [1, 3, 0, 2, 0, 2]
startState = [7, 7]

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

baseParameters['sensor']['explorer'] = yaml.dump(["regions.secondExplorer", {'start' : startState,
                                                                             'length' : SEQ_SIZE,
                                                                             'step' : STEP}])

print '\nstart, actions = ', startState, actions

net = createNetwork(baseParameters)
explorer = net.regions['sensor'].getSelf().explorer[2]
explorer.setMoveList(actions)

trainLosses = []
testLosses = []

for i in range(3):
    acc, epochClsLosses = train(net, 'mnist', SEQ_SIZE)
    trainLosses.append(epochClsLosses)
    print 'Train acc:\t%3.1f' % acc

    acc, first, testEpochLoss = test(net, 'mnist', SEQ_SIZE)
    testLosses.append(testEpochLoss)
    print 'Test acc:\t%3.1f' % (acc)
    print 'bystep:\t', first

with open('test_losses.txt', 'w') as f:
    print >> f, testLosses

with open('train_losses.txt', 'w') as f:
    for epoch in trainLosses:
        for item in epoch:
            f.write('%f,' % item)
        f.write('\n')