from updaters import *
from network import *
import yaml
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

model = CategoryModel(10, epsilon=0.2)

net = createNetwork(baseParameters)
sensor = net.regions['sensor']
classifier = net.regions['CLS']
explorer = sensor.getSelf().explorer[2]

for i in range(10):
    acc, first, second = modifiedTrain(net, model, startState, SEQ_SIZE, 'mnist')
    print 'Train acc:\t%3.1f' % acc
    print '\tPrime:\t', first.astype(np.int)
    print '\tSecond:\t', second.astype(np.int)

    acc, first, second = modifiedTest(net, model, startState, SEQ_SIZE, 'mnist')
    print 'Test acc:\t%3.1f' % acc
    print '\tPrime:\t', first.astype(np.int)
    print '\tSecond:\t', second.astype(np.int)
    print '\n'
    break

exit(0)


sensor.executeCommand(['loadMultipleImages', 'mnist/small_testing'])
numImages = sensor.getParameter('numImages')
classifier.setParameter('inferenceMode', 1)
net.regions['SP2'].setParameter('inferenceMode', 1)
net.regions['TM'].setParameter('inferenceMode', 1)
net.regions['SP'].setParameter('inferenceMode', 1)

net.initialize()

for i in range(numImages):

    explorer.setMoveList([])
    for j in range(SEQ_SIZE):
        net.run(1)
        sens_out = sensor.getOutputData('dataOut').reshape(10, 10)
        print explorer.position
        for s in sens_out:
            print ''.join('_' if e == 0 else '&' for e in s)
        position = explorer.position['offset']
        catVec = classifier.getOutputData("categoriesOut")
        print catVec
        currentCategory = np.random.choice(np.arange(0, 10), p=catVec)
        action = model.getNextAction(currentCategory, position)
        explorer.addAction(action)


    catVec = classifier.getOutputData("categoriesOut")
    inferredCategory = catVec.argmax()
    print inferredCategory










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