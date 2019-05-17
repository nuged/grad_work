import yaml
from nupic.engine import Network
from time import time, sleep
import numpy as np
from regions import *
import os, sys
import math

class myNetwork(Network):

    def run(self, n, log):
        sensor = self.regions['sensor']
        sp = self.regions['SP']
        tm = self.regions['TM']
        sp2 = self.regions['SP2']
        cls = self.regions['CLS']

        if log:
            f = open('logs.txt', 'a')
            orig = sys.stdout
            sys.stdout = f

            image = sensor.getSelf()._getOriginalImage().split()[0]
            image = np.array(image)
            for s in image:
                print ''.join('_' if e == 0 else '&' for e in s)

            sys.stdout = orig
            f.close()
        sp2Learning = sp2.getParameter('learningMode')
        sp2Inference = sp2.getParameter('inferenceMode')
        clsLearning = cls.getParameter('learningMode')
        clsInference = cls.getParameter('inferenceMode')

        cls.setParameter('learningMode', 0)
        cls.setParameter('inferenceMode', 0)
        sp2.setParameter('learningMode', 0)
        sp2.setParameter('inferenceMode', 0)

        for i in range(n):
            if i == n - 1 and clsLearning:
                cls.setParameter('learningMode', 1)
            if i == n - 1 and sp2Learning:
                sp2.setParameter('learningMode', 1)
            if i == n - 1 and sp2Inference:
                sp2.setParameter('inferenceMode', 1)
            if i == n - 1 and clsInference:
                cls.setParameter('inferenceMode', 1)

            if log:
                Network.run(self, 1)

                f = open('logs.txt', 'a')
                orig = sys.stdout
                sys.stdout = f

                print '\n\nsensor'
                print sensor.getSelf().explorer[2].position
                print sensor.getOutputData('resetOut')
                sens_out = sensor.getOutputData('dataOut').reshape(10, 10)
                for s in sens_out:
                    print ''.join('_' if e == 0 else '&' for e in s)

                #print '\nsp1'
                #sp1_in = sp.getInputData('bottomUpIn').reshape(10, 10)
                #for s in sp1_in:
                #    print ''.join('_' if e == 0 else '&' for e in s)
                #print sp.getInputData('resetIn')
                #sp1_out = sp.getOutputData('bottomUpOut').reshape(32, 128)
                #for s in sp1_out:
                #    print ''.join('_' if e == 0 else '&' for e in s)

                #print '\ntm'
                #print np.all(tm.getInputData('bottomUpIn') == sp1_out.reshape(-1))
                #print tm.getInputData('resetIn')
                #tm_bu = tm.getOutputData('bottomUpOut').reshape(2, 50, 160)

                #print 'bottomUp'
                #for mat in tm_bu:
                #    for s in mat:
                #        print ''.join('_' if e == 0 else '&' for e in s)
                #    print '\n'

                #print 'sp2'
                #print np.all(sp2.getInputData('bottomUpIn') == tm_bu.reshape(-1))
                #print sp2.getInputData('resetIn')
                #sp2_out = sp2.getOutputData('bottomUpOut').reshape(16, 128)
                #for s in sp2_out:
                #    print ''.join('_' if e == 0 else '&' for e in s)

                #print '\ncls'
                #print np.all(cls.getInputData('bottomUpIn') == sp2_out.reshape(-1))
                #print 'catOut:', cls.getOutputData('categoriesOut')

                sys.stdout = orig
                f.close()
            else:
                Network.run(self, 1)


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

    net.addRegion('SP2', 'py.mySP', yaml.dump(SP2_PARAMS))

    Network.registerRegion(myClassifier)
    net.addRegion("CLS", "py.myClassifier", yaml.dump(CLS_PARAMS))

    net.link("sensor", "SP", "UniformLink", "",
             srcOutput="dataOut", destInput="bottomUpIn")

    net.link("sensor", "SP", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "TM", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")
    net.link("sensor", "SP2", "UniformLink", "",
             srcOutput="resetOut", destInput="resetIn")

    net.link("SP", "TM", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")

    net.link("TM", "SP2", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")

    net.link("SP2", "CLS", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="bottomUpIn")

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

    print '\nnPatterns learned:', classifier.getParameter('patternCount')


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
        trueCategory = sensor.getOutputData("categoryOut")
        #print catVec, trueCategory
        idx = int(trueCategory[0])
        catVec[idx] -= 1
        #print catVec
        se = np.square(catVec).sum()
        ses.append(se)
        if sensor.getOutputData("categoryOut") == inferredCategory:
            numCorrect += 1
        if i % every == every - 1:
            print "\t%d-th iteration, nCorrect=%d" % (i, numCorrect)
    mse = np.mean(ses)

    return (100.0 * numCorrect) / numTestImages, mse
