from network import *
import yaml
from time import time

IMG_SIZE = 28
WINDOW_SIZE = 10
STEP = 7
SEQ_SIZE = 6    # num of positions
actions = [1, 3, 0, 2, 1]
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

trainAccs = []
testAccs = []

for i in range(5):
    acc, epochClsLosses = train(net, 'mnist', SEQ_SIZE)
    trainLosses.append(epochClsLosses)
    trainAccs.append(acc)
    print 'Train acc:\t%3.1f' % acc

    acc, first, testEpochLoss = test(net, 'mnist', SEQ_SIZE)
    testLosses.append(testEpochLoss)
    testAccs.append(acc)
    print 'Test acc:\t%3.1f' % (acc)
    print 'bystep:\t', first

FILE_PREFIX = 'scores/%s/' % ''.join([str(a) for a in actions])

if not os.path.exists(FILE_PREFIX):
    os.mkdir(FILE_PREFIX)

with open(FILE_PREFIX + 'test_losses.txt', 'w') as f:
    print >> f, testLosses

with open(FILE_PREFIX + 'train_accs.txt', 'w') as f:
    print >> f, trainAccs

with open(FILE_PREFIX + 'test_accs.txt', 'w') as f:
    print >> f, testAccs

with open(FILE_PREFIX + 'train_losses.txt', 'w') as f:
    for epoch in trainLosses:
        for item in epoch:
            f.write('%f,' % item)
        f.write('\n')