import yaml
from nupic.engine import Network
from time import time, sleep
import numpy as np
from regions import *
import os, sys
import math
from utils import *

class myNetwork(Network):

    def run(self, n, log=False):
        sensor = self.regions['sensor']
        sp = self.regions['SP']
        tm = self.regions['TM']
        sp2 = self.regions['SP']
        cls = self.regions['CLS']

        Network.run(self, n)

        if log:

            f = open('logs.txt', 'a')
            orig = sys.stdout
            sys.stdout = f

            print '\n\nsensor'
            print sensor.getSelf().explorer[2].position
            #print sensor.getSelf().getSequenceCount()
            print 'RESET:', sensor.getOutputData('resetOut')
            sens_out = sensor.getOutputData('dataOut').reshape(10, 10)
            for s in sens_out:
                print ''.join('_' if e == 0 else '&' for e in s)

            #print '\nsp1'
            sp1_in = sp.getInputData('bottomUpIn').reshape(10, 10)
            #for s in sp1_in:
            #   print ''.join('_' if e == 0 else '&' for e in s)
            #print 'RESET:', sp.getInputData('resetIn')
            sp1_out = sp.getOutputData('bottomUpOut').reshape(32, 128)
            #for s in sp1_out:
            #    print ''.join('_' if e == 0 else '&' for e in s)

            #print '\ntm'
            #print np.all(tm.getInputData('bottomUpIn') == sp1_out.reshape(-1))
            #print 'RESET:', tm.getInputData('resetIn')
            tm_bu = tm.getOutputData('bottomUpOut').reshape(2, 32, 128)

            #print 'bottomUp'
            #for mat in tm_bu:
            #   for s in mat:
            #       print ''.join('_' if e == 0 else '&' for e in s)
            #   print '\n'

            #print 'sp2'
            #print np.all(sp2.getInputData('bottomUpIn') == tm_bu.reshape(-1))
            #print 'RESET:', sp2.getInputData('resetIn')
            sp2_out = sp2.getOutputData('bottomUpOut').reshape(16, 128)
            #for s in sp2_out:
            #    print ''.join('_' if e == 0 else '&' for e in s)
            print '\n'
            #print '\ncls'
            #print np.all(cls.getInputData('bottomUpIn') == sp2_out.reshape(-1))
            #print 'catOut:', cls.getOutputData('categoriesOut')

            sys.stdout = orig
            f.close()

def createNetwork(params):
    IMAGE_SENSOR_PARAMS = params['sensor']
    SP_PARAMS = params['SP']
    SP2_PARAMS = params['SP2']
    TM_PARAMS = params['TM']
    CLS_PARAMS = params['CLS']

    net = myNetwork()

    Network.registerRegion(mySensor)
    net.addRegion('sensor', 'py.mySensor', yaml.dump(IMAGE_SENSOR_PARAMS))

    Network.registerRegion(mySP)
    net.addRegion('SP', 'py.mySP', yaml.dump(SP_PARAMS))

    net.addRegion('TM', 'py.TMRegion', yaml.dump(TM_PARAMS))

    #net.addRegion('SP2', 'py.mySP', yaml.dump(SP2_PARAMS))

    Network.registerRegion(myClassifier)
    net.addRegion("CLS", "py.myClassifier", yaml.dump(CLS_PARAMS))

    net.link("sensor", "SP", "UniformLink", "",
             srcOutput="dataOut", destInput="bottomUpIn")

    net.link("sensor", "SP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    #net.link("sensor", "SP2", "UniformLink", "",
    #         srcOutput="resetOut", destInput="resetIn")

    net.link("SP", "TM", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")

    net.link("TM", "CLS", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")

    #net.link("SP2", "CLS", "UniformLink", "",
    #         srcOutput="bottomUpOut", destInput="bottomUpIn")

    net.link("sensor", "CLS", "UniformLink", "",
             srcOutput="categoryOut", destInput="categoryIn")

    return net


def train(net, dataDir, fullSample=False):
    sensor = net.regions["sensor"]
    sp = net.regions["SP"]
    tm = net.regions['TM']
    sp2 = net.regions['SP2']
    classifier = net.regions["CLS"]


    imgIterations = sensor.getSelf().explorer[2].getImageIterations()

    if fullSample:
        path = os.path.join(dataDir, "training")
    else:
        path = os.path.join(dataDir, 'small_training')

    start = time()
    sensor.executeCommand(["loadMultipleImages", path])
    numTrainingImages = sensor.getParameter("numImages")
    end = time()

    print 'Loaded %d training samples in %3.2f seconds' % (numTrainingImages, (end-start))

    # ----------------------------------------Phase 1----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 1)
    sp.setParameter("inferenceMode", 0)
    tm.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 0)

    nTrainingIterations = numTrainingImages
    print "---Phase 1---"
    start = time()
    for i in range(nTrainingIterations):
        net.run(imgIterations, False)

    print '\tFinished in %06.2f sec' % (time() - start)

    # ----------------------------------------Phase 2----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 1)
    tm.setParameter("inferenceMode", 0)
    sp2.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 0)
    nTrainingIterations = numTrainingImages

    print "---Phase 2---"
    start = time()
    for i in range(nTrainingIterations):
        net.run(imgIterations, False)
    print '\tFinished in %06.2f sec' % (time() - start)

    # ----------------------------------------Phase 3----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 1)
    sp2.setParameter("inferenceMode", 0)

    nTrainingIterations = numTrainingImages
    print "---Phase 3---"
    start = time()
    for i in range(nTrainingIterations):
        net.run(imgIterations, False)
    print '\tFinished in %06.2f sec' % (time() - start)

    # ----------------------------------------Classifier TRAINING----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 1)
    sp.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)

    print "---CLS training---"
    start = time()
    for i in range(nTrainingIterations):
        net.run(imgIterations, False)
    print '\tFinished in %06.2f sec' % (time() - start)

    return classifier.getParameter('patternCount')


def test(net, dataDir, fullSample=False):
    sensor = net.regions["sensor"]
    sp = net.regions["SP"]
    sp2 = net.regions['SP2']
    tm = net.regions['TM']
    classifier = net.regions["CLS"]

    imgIterations = sensor.getSelf().explorer[2].getImageIterations()

    if fullSample:
        path = os.path.join(dataDir, "testing")
    else:
        path = os.path.join(dataDir, 'small_testing')

    start = time()
    sensor.executeCommand(["loadMultipleImages", path])
    numTestImages = sensor.getParameter("numImages")
    end = time()

    print '\nLoaded %d testing samples in %3.2f seconds' % (numTestImages, (end-start))

    classifier.setParameter("inferenceMode", 1)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    sp.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 0)

    print('---Testing---')
    numCorrect = 0

    every = numTestImages // 10
    ses = []
    for i in range(numTestImages):
        if i == 48:
            net.run(imgIterations, False)
        else:
            net.run(imgIterations, False)
        catVec = classifier.getOutputData("categoriesOut")
        inferredCategory = catVec.argmax()

        if sensor.getOutputData("categoryOut") == inferredCategory:
            numCorrect += 1
        if i % every == every - 1:
            print "\t%d-th iteration, nCorrect=%d" % (i, numCorrect)

    return (100.0 * numCorrect) / numTestImages


def modifiedTrain(net, model, startPosition, length, dataDir, fullSample=False):
    sensor = net.regions["sensor"]
    explorer = sensor.getSelf().explorer[2]
    sp = net.regions["SP"]
    tm = net.regions['TM']
    sp2 = net.regions['SP']
    classifier = net.regions["CLS"]

    if fullSample:
        path = os.path.join(dataDir, "training")
    else:
        path = os.path.join(dataDir, 'small_training')

    start = time()
    sensor.executeCommand(["loadMultipleImages", path])
    numTrainingImages = sensor.getParameter("numImages")
    end = time()
    print 'Loaded %d training samples in %3.2f seconds' % (numTrainingImages, (end - start))

    net.initialize()

    sp.setParameter("inferenceMode", 1)
    sp.setParameter("learningMode", 1)
    tm.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 1)
    sp2.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 1)
    classifier.setParameter("inferenceMode", 1)
    classifier.setParameter("learningMode", 1)

    print "---Training---"
    start = time()
    numCorrect = 0
    correctByStamp = np.zeros(length)
    secondByStamp = np.zeros(length)

    for i in range(numTrainingImages):
        explorer.setMoveList([])
        #classifier.setParameter("learningMode", 0)
        if i in [22, 23]:
            log = False
        else:
            log = False
        for j in range(length):
            if j == length - 1:
                pass #classifier.setParameter("learningMode", 1)

            net.run(1, log)
            #print sensor.getSelf().explorer[2].position
            currentCategory = int(sensor.getOutputData('categoryOut')[0])
            #probs = classifier.getOutputData('probabilities')
            probs = classifier.getOutputData('categoryProbabilitiesOut')
            firstVal = probs.argmax()
            if firstVal == currentCategory:
                correctByStamp[j] += 1

            secondVal = probs.argsort()[-2]
            if secondVal == currentCategory:
                secondByStamp[j] += 1


            if j == 0:
                #print startPosition, currentCategory, length, 'before'
                sequence = model.createSequence(currentCategory, copy.deepcopy(startPosition), length)
                #print startPosition, currentCategory, length, 'after'
                explorer.setMoveList(sequence)

            sensor.getSelf().explorer[2].customNext()

            #print catVec
            #for state, act in model.stateActionSequence:
            #    print getOffset(state, 4, 7), act, state
            #print sensor.getOutputData("categoryOut"), catVec.argmax(), explorer.position, explorer.moveList, '\n'

        inferredCategory = probs.argmax()
        if inferredCategory == currentCategory:
            model.update(currentCategory, 15)
            numCorrect += 1
        else:
            model.update(currentCategory, -1)
    print '\tFinished in %06.2f sec' % (time() - start)

    return 100. * numCorrect / numTrainingImages, 100. * correctByStamp / numTrainingImages,\
           100. * secondByStamp / numTrainingImages


def modifiedTest(net, model, startPosition, length, dataDir, fullSample=False):
    sensor = net.regions["sensor"]
    explorer = sensor.getSelf().explorer[2]
    sp = net.regions["SP"]
    tm = net.regions['TM']
    sp2 = net.regions['SP']
    classifier = net.regions["CLS"]

    if fullSample:
        path = os.path.join(dataDir, "testing")
    else:
        path = os.path.join(dataDir, 'small_testing')

    start = time()
    sensor.executeCommand(["loadMultipleImages", path])
    numTestImages = sensor.getParameter("numImages")
    end = time()
    print 'Loaded %d testing samples in %3.2f seconds' % (numTestImages, (end - start))

    net.initialize()
    explorer.first()
    classifier.setParameter("inferenceMode", 1)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    sp.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 0)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print('---Testing---')
    numCorrect = 0
    #np.random.seed(42)
    every = numTestImages + 100
    first = np.zeros(length)
    second = np.zeros(length)

    positions = np.zeros((numTestImages, length))

    q = 4

    for i in range(numTestImages):
        explorer.setMoveList([])
        if i in [22, 23]:
            log = False
        else:
            log = False
        for j in range(length):
            #print explorer.position, explorer.moveList
            net.run(1, log)
            #print sensor.getSelf().explorer[2].position
            currentCategory = int(sensor.getOutputData('categoryOut')[0])
            #probs = classifier.getOutputData('probabilities')
            probs = classifier.getOutputData('categoryProbabilitiesOut')
            currentPosition = sensor.getSelf().explorer[2].position['offset']

            if j == 1:
                q = 5

            if j == 2:
                q = 3

            if j == 4:
                q = 2


            qval = np.sort(probs)[::-1][q]
            categories = np.nonzero(probs >= qval)[0]

            action = model.getNextAction(categories, probs[categories], copy.deepcopy(currentPosition))

            explorer.addAction(action)

            positions[i][j] = np.nonzero(currentCategory == probs.argsort())[0][0]

            firstVal = probs.argmax()
            if firstVal == currentCategory:
                first[j] += 1

            secondVal = probs.argsort()[-2]
            if secondVal == currentCategory:
                second[j] += 1

            #if j == 0:
           #     sequence = model.createSequence(currentCategory, copy.deepcopy(startPosition), length)
                # print startPosition, currentCategory, length, 'after'
           #     explorer.setMoveList(sequence)

            sensor.getSelf().explorer[2].customNext()

            #position = explorer.position['offset']

            #currentCategory = int(sensor.getOutputData("categoryOut")[0])
            #action = model.getNextAction(currentCategory, copy.deepcopy(position))

            #explorer.addAction(action)
            #print catVec
            #print model.stateActionSequence
            #print sensor.getOutputData("categoryOut"), catVec.argmax(), explorer.position, explorer.moveList, '\n'
        catVec = probs
        if sensor.getOutputData("categoryOut") == catVec.argmax():
            numCorrect += 1

        if i % every == every - 1:
            print "\t%d-th iteration, nCorrect=%d" % (i, numCorrect)

    return (100.0 * numCorrect) / numTestImages, positions