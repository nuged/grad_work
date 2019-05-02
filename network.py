import yaml
from nupic.engine import Network
from time import time, sleep
import numpy as np
from regions import *
import os, sys

class myNetwork(Network):

    def run(self, n, log):
        sensor = self.regions['sensor']
        sp = self.regions['SP']
        tm = self.regions['TM']
        sp2 = self.regions['SP2']
        cls = self.regions['CLS']

        clsLearning = cls.getParameter('learningMode')
        cls.setParameter('learningMode', 0)

        for i in range(n):
            if i == n - 1 and clsLearning:
                cls.setParameter('learningMode', 1)
            Network.run(self, 1)

            if log:
                f = open('logs.txt', 'a')
                orig = sys.stdout
                sys.stdout = f

                print '\n\nsensor'
                sens_out = sensor.getOutputData('dataOut').reshape(32, 32)
                for s in sens_out:
                    print ''.join('_' if e == 0 else '&' for e in s)

                print '\nsp1'
                sp1_out = sp.getOutputData('bottomUpOut').reshape(32, 128)
                for s in sp1_out:
                    print ''.join('_' if e == 0 else '&' for e in s)

                print '\ntm'
                tm_bu = tm.getOutputData('bottomUpOut').reshape(8, 16, 128)

                print 'bottomUp'
                for mat in tm_bu:
                    for s in mat:
                        print ''.join('_' if e == 0 else '&' for e in s)
                    print '\n'

                print 'sp2'
                sp2_out = sp2.getOutputData('bottomUpOut').reshape(16, 128)
                for s in sp2_out:
                    print ''.join('_' if e == 0 else '&' for e in s)

                print '\ncls'
                print 'catOut:', cls.getOutputData('categoriesOut')
                print 'probs:', cls.getOutputData('categoryProbabilitiesOut')

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


def train(net, dataDir):
    sensor = net.regions["sensor"]
    sp = net.regions["SP"]
    sp2 = net.regions['SP2']
    tm = net.regions['TM']
    classifier = net.regions["CLS"]

    imgIterations = sensor.getSelf().explorer[2].getImageIterations()

    t1 = time()
    sensor.executeCommand(["loadMultipleImages", os.path.join(dataDir, "small_training")])
    numTrainingImages = sensor.getParameter("numImages")
    start = time()
    seconds = (start - t1)
    print "Load time for training images:\t%05.2f sec" % seconds
    print "Number of training images", numTrainingImages

    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 1)
    sp.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 1)
    sp2.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 1)
    tm.setParameter("inferenceMode", 1)

    nTrainingIterations = numTrainingImages
    print "---HTM training---"
    with open('logs.txt', 'w') as f:
        f.write('-----HTM Training-----\n')
    start = time()
    for i in range(nTrainingIterations):
        if i % 1000 == 999:
            t1 = time()
            mins = (t1 - start) / 60
            print "\t%d-th iteration, %05.2f min" % (i, mins)
        if i % 1000 == 0:
            log = False
        else:
            log = False
        net.run(imgIterations, log)

    classifier.setParameter("inferenceMode", 1)
    classifier.setParameter("learningMode", 1)
    sp.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)

    print "---CLS training---"
    with open('logs.txt', 'a') as f:
        f.write('\n-----CLS Training-----\n')
    start = time()
    for i in range(nTrainingIterations):
        if i % 1000 == 999:
            t1 = time()
            mins = (t1 - start) / 60
            print "\t%d-th iteration, %05.2f min" % (i, mins)
        if i % 1000 == 0:
            log = False
        else:
            log = False
        net.run(imgIterations, log)



def test(net, dataDir):
    sensor = net.regions["sensor"]
    sp = net.regions["SP"]
    sp2 = net.regions['SP2']
    tm = net.regions['TM']
    classifier = net.regions["CLS"]

    imgIterations = sensor.getSelf().explorer[2].getImageIterations()

    sensor.executeCommand(["loadMultipleImages", os.path.join(dataDir, "small_testing")])

    numTestImages = sensor.getParameter("numImages")

    classifier.setParameter("inferenceMode", 1)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    sp.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 0)
    print('---Testing---')
    with open('logs.txt', 'a') as f:
        f.write('\n-----Testing-----\n')
    numCorrect = 0
    for i in range(numTestImages):
        if i % 1000 == 0:
            log = False
        else:
            log = False
        net.run(imgIterations, log)
        inferredCategory = classifier.getOutputData("categoriesOut").argmax()
        if sensor.getOutputData("categoryOut") == inferredCategory:
            numCorrect += 1
        if i % 100 == 99:
            print "\t%d-th iteration, nCorrect=%d" % (i, numCorrect)

    return (100.0 * numCorrect) / numTestImages
