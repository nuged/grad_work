import yaml
from nupic.engine import Network
from time import time, sleep
import numpy as np
from regions import *
import os, sys, utils
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
    AE_PARAMS = params['AutoEncoder']
    CLS_PARAMS = params['Net']

    net = Network()

    Network.registerRegion(mySensor)
    net.addRegion('sensor', 'py.mySensor', yaml.dump(IMAGE_SENSOR_PARAMS))

    Network.registerRegion(mySP)
    net.addRegion('SP', 'py.mySP', yaml.dump(SP_PARAMS))

    Network.registerRegion(AERegion)
    net.addRegion('AE', 'py.AERegion', yaml.dump(AE_PARAMS))

    Network.registerRegion(myClassifier)
    net.addRegion("CLS", "py.myClassifier", yaml.dump(CLS_PARAMS))

    net.link("sensor", "SP", "UniformLink", "",
             srcOutput="dataOut", destInput="bottomUpIn")

    net.link("SP", "AE", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")

    net.link("AE", "CLS", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")

    net.link("sensor", "CLS", "UniformLink", "",
             srcOutput="categoryOut", destInput="categoryIn")

    return net


def train(net, dataDir, length, fullSample=False):
    sensor = net.regions["sensor"]
    sp = net.regions["SP"]
    ae = net.regions['AE']
    classifier = net.regions["CLS"]
    explorer = sensor.getSelf().explorer[2]

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
    net.initialize()
    classifier.setParameter("inferenceMode", 1)
    #classifier.setParameter("learningMode", 1)
    sp.setParameter("learningMode", 1)
    sp.setParameter("inferenceMode", 1)
    ae.setParameter("learningMode", 1)
    ae.setParameter("inferenceMode", 1)

    print "---Training---"
    numCorrect = 0
    start = time()
    for i in range(numTrainingImages):
        classifier.setParameter("learningMode", 0)
        if i in [22, 23]:
            log = False
        else:
            log = False
        for j in range(length):
            if j == length - 1:
                classifier.setParameter("learningMode", 1)

            net.run(1)

            #print utils.getIndex(explorer.position['offset'], 4, 7)

            explorer.customNext()

        probs = classifier.getOutputData('probabilities')
        if probs.argmax() == sensor.getOutputData('categoryOut'):
            numCorrect += 1
    print 'Took %5.2f sec.' % (time() - start)
    pycls = classifier.getSelf()
    losses = copy.copy(pycls.lossValues)
    pycls.lossValues = []
    pycls.loss_idx = 0
    return numCorrect * 100. / numTrainingImages, losses


def test(net, dataDir, length, fullSample=False):
    sensor = net.regions["sensor"]
    sp = net.regions["SP"]
    ae = net.regions['AE']
    classifier = net.regions["CLS"]

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
    ae.setParameter("inferenceMode", 1)
    ae.setParameter("learningMode", 0)

    print('---Testing---')
    numCorrect = 0
    first = np.zeros(length)
    for i in range(numTestImages):
        if i in [22, 23]:
            log = False
        else:
            log = False
        for j in range(length):

            net.run(1)
            probs = classifier.getOutputData('probabilities')
            currentCategory = int(sensor.getOutputData('categoryOut')[0])

            firstVal = probs.argmax()
            if firstVal == currentCategory:
                first[j] += 1

            sensor.getSelf().explorer[2].customNext()


        if probs.argmax() == sensor.getOutputData('categoryOut'):
            numCorrect += 1
    pycls = classifier.getSelf()
    losses = copy.copy(pycls.lossValues)
    pycls.lossValues = []
    pycls.loss_idx = 0
    return (100.0 * numCorrect) / numTestImages, 100. * first / numTestImages, np.mean(losses)


def modifiedTrain(net, model, startPosition, length, dataDir, fullSample=False):
    sensor = net.regions["sensor"]
    explorer = sensor.getSelf().explorer[2]
    sp = net.regions["SP"]
    ae = net.regions['AE']
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
    ae.setParameter("inferenceMode", 1)
    ae.setParameter("learningMode", 1)
    classifier.setParameter("inferenceMode", 1)
    #classifier.setParameter("learningMode", 1)

    print "---Training---"
    start = time()
    numCorrect = 0
    correctByStamp = np.zeros(length)

    for i in range(numTrainingImages):
        explorer.setMoveList([])
        classifier.setParameter("learningMode", 0)
        if i in [22, 23]:
            log = False
        else:
            log = False

        numEmpty = 0

        for j in range(length):
            if j == length - 1:
                classifier.setParameter("learningMode", 1)

            net.run(1)

            numEmpty += np.count_nonzero(sensor.getOutputData('dataOut')) < 15

            currentCategory = int(sensor.getOutputData('categoryOut')[0])
            probs = classifier.getOutputData('probabilities')
            #probs = classifier.getOutputData('categoryProbabilitiesOut')

            firstVal = probs.argmax()
            if firstVal == currentCategory:
                correctByStamp[j] += 1

            if j == 0:
                sequence = model.createSequence(currentCategory, copy.deepcopy(startPosition), length)
                explorer.setMoveList(sequence)

            sensor.getSelf().explorer[2].customNext()

        states = np.array(model.stateActionSequence[::-1])[:, 0]
        uniqueStates = np.unique(states)
        numUnique = len(uniqueStates)

        if probs.argmax() == currentCategory:
            numCorrect += 1
            if numEmpty > 1:
                model.update(currentCategory, 5)
            else:
                if numUnique >= 5:
                    model.update(currentCategory, 100)
                elif numUnique == 4:
                    model.update(currentCategory, 50)
                else:
                    model.update(currentCategory, 25)
        else:
            model.update(currentCategory, -1)

    print '\tFinished in %06.2f sec' % (time() - start)
    pycls = classifier.getSelf()
    losses = copy.copy(pycls.lossValues)
    pycls.lossValues = []
    pycls.loss_idx = 0
    return 100. * numCorrect / numTrainingImages, 100. * correctByStamp / numTrainingImages, losses


def modifiedTest(net, model, startPosition, length, dataDir, fullSample=False):
    sensor = net.regions["sensor"]
    explorer = sensor.getSelf().explorer[2]
    sp = net.regions["SP"]
    ae = net.regions['AE']
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
    ae.setParameter("inferenceMode", 1)
    ae.setParameter("learningMode", 0)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    print('---Testing---')
    numCorrect = 0
    every = numTestImages + 100
    first = np.zeros(length)
    positions = np.zeros((numTestImages, length))

    q = 2
    #32-32-33.5
    nKeep = 1

    categoryPaths = np.zeros((10, length, numTestImages // 10))
    listEmpty = []
    for i in range(numTestImages):
        explorer.setMoveList([])
        if i in [22, 23]:
            log = False
        else:
            log = False

        numEmpty = 0
        for j in range(length):
            net.run(1)

            numEmpty += np.count_nonzero(sensor.getOutputData('dataOut')) < 15

            currentCategory = int(sensor.getOutputData('categoryOut')[0])
            probs = classifier.getOutputData('probabilities')
            #probs = classifier.getOutputData('categoryProbabilitiesOut')
            currentPosition = sensor.getSelf().explorer[2].position['offset']
            #print currentCategory, explorer.position, sensor.getOutputData('resetOut')

            if j == 1:
                q = 4
            if j == 2:
                q = 3
            if j == 4:
                q = 1
            qval = np.sort(probs)[::-1][q]
            categories = np.nonzero(probs >= qval)[0]

            action = model.getNextAction(categories, probs[categories], copy.deepcopy(currentPosition))

            state = utils.getIndex(copy.deepcopy(currentPosition), 4, 7)
            categoryPaths[currentCategory, j, i // 10] = state

            if j == 0:
                bestCategories = probs.argsort()[::-1][:nKeep]
                bestProbs = probs[bestCategories]
                bestProbs = np.exp(bestProbs) / sum(np.exp(bestProbs))
                category = np.random.choice(bestCategories, p=bestProbs)
                sequence = model.createSequence(category, copy.deepcopy(startPosition), length,
                                                random=False, store=False)
                explorer.setMoveList(sequence)

            #explorer.addAction(action)

            positions[i][j] = np.nonzero(currentCategory == probs.argsort())[0][0]

            firstVal = probs.argmax()
            if firstVal == currentCategory:
                first[j] += 1

            sensor.getSelf().explorer[2].customNext()
        #print 'numEmpty:', numEmpty
        listEmpty.append(numEmpty)
        if sensor.getOutputData("categoryOut") == probs.argmax():
            numCorrect += 1

        if i % every == every - 1:
            print "\t%d-th iteration, nCorrect=%d" % (i, numCorrect)
        pycls = classifier.getSelf()
        losses = copy.copy(pycls.lossValues)
        pycls.lossValues = []
        pycls.loss_idx = 0
    return (100.0 * numCorrect) / numTestImages, 100. * first / numTestImages, categoryPaths, np.mean(losses)