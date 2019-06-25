from network import *
import yaml
from reinforcement import CategoryModel

IMG_SIZE = 28
WINDOW_SIZE = 10
STEP = 7
SEQ_SIZE = 6    # num of positions
startState = [7, 7]

with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

baseParameters['SP']['inputWidth'] = WINDOW_SIZE * WINDOW_SIZE
baseParameters['sensor']['width'] = WINDOW_SIZE
baseParameters['sensor']['height'] = WINDOW_SIZE

baseParameters['sensor']['explorer'] = yaml.dump(["regions.secondExplorer", {'start' : startState,
                                                                             'length' : SEQ_SIZE,
                                                                             'step' : STEP}])

model = CategoryModel(10, epsilon=0.1, ignoreCategory=False)

net = createNetwork(baseParameters)
sensor = net.regions['sensor']
classifier = net.regions['CLS']
explorer = sensor.getSelf().explorer[2]

trainLosses = []
testLosses = []

trainAccs = []
testAccs = []

for i in range(3):
    acc, first, epochClsLosses = modifiedTrain(net, model, startState, SEQ_SIZE, 'mnist')
    print 'Train acc:\t%3.1f' % acc, first
    # single - 74.9, 82.1, 81.9, 80.2
    # parallel - 49, 49.5, 49.1, 52.2
    # true path - 88.8, 95, 85.9, 96.1
    acc, first, testEpochLoss, answers = parallelTest(net, model, startState, SEQ_SIZE, 'mnist')
    print 'Test acc:\t', testEpochLoss, 'first acc:\t', first, 'by cat:\t', acc

    print answers

    for j in range(10):
        actPath = model.createSequence(j, startState, SEQ_SIZE, random=False, store=True)
        path = []
        for state, act in model.stateActionSequence[::-1]:
            path.append(state)
        print '\'%d\':\t' % j, path, ','
        #print '\tres:\t', pos.mean(axis=2)[j][:-1]
    print '\n'

FILE_PREFIX = 'scores/MC/'

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

path = [5, 6, 10, 9, 8, 9]