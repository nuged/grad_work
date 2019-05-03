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

        clsLearning = cls.getParameter('learningMode')
        cls.setParameter('learningMode', 0)

        for i in range(n):
            if i == n - 1 and clsLearning:
                cls.setParameter('learningMode', 1)
            if log:
                Network.run(self, 1)

                f = open('logs.txt', 'a')
                orig = sys.stdout
                sys.stdout = f

                print '\n\nsensor'
                #print sensor.getSelf().explorer[2].position
                print sensor.getOutputData('resetOut')
                sens_out = sensor.getOutputData('dataOut').reshape(20, 20)
                for s in sens_out:
                    print ''.join('_' if e == 0 else '&' for e in s)

                print '\nsp1'
                sp1_in = sp.getInputData('bottomUpIn').reshape(20, 20)
                #for s in sp1_in:
                #    print ''.join('_' if e == 0 else '&' for e in s)
                print sp.getInputData('resetIn')
                sp1_out = sp.getOutputData('bottomUpOut').reshape(32, 128)
                for s in sp1_out:
                    print ''.join('_' if e == 0 else '&' for e in s)

                print '\ntm'
                #print np.all(tm.getInputData('bottomUpIn') == sp1_out.reshape(-1))
                print tm.getInputData('resetIn')
                tm_bu = tm.getOutputData('bottomUpOut').reshape(4, 16, 128)

                print 'bottomUp'
                for mat in tm_bu:
                    for s in mat:
                        print ''.join('_' if e == 0 else '&' for e in s)
                    print '\n'

                print 'sp2'
                #print np.all(sp2.getInputData('bottomUpIn') == tm_bu.reshape(-1))
                print sp2.getInputData('resetIn')
                sp2_out = sp2.getOutputData('bottomUpOut').reshape(16, 128)
                for s in sp2_out:
                    print ''.join('_' if e == 0 else '&' for e in s)

                print '\ncls'
                #print np.all(cls.getInputData('bottomUpIn') == sp2_out.reshape(-1))
                print 'catOut:', cls.getOutputData('categoriesOut')

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


def train(net, dataDir):
    sensor = net.regions["sensor"]
    sp = net.regions["SP"]
    tm = net.regions['TM']
    sp2 = net.regions['SP2']
    classifier = net.regions["CLS"]

    imgIterations = sensor.getSelf().explorer[2].getImageIterations()

    sensor.executeCommand(["loadMultipleImages", os.path.join(dataDir, "small_training")])
    numTrainingImages = sensor.getParameter("numImages")

    # ----------------------------------------SP1 TRAINING----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 1)
    sp.setParameter("inferenceMode", 0)
    tm.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 0)

    nTrainingIterations = numTrainingImages
    print "---SP1 training---"
    start = time()
    for i in range(nTrainingIterations):
        net.run(imgIterations, False)
    print 'Finished in %06.2f sec' % (time() - start)

    '''
    # ----------------------------------------TM TRAINING----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 1)
    tm.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 0)

    nTrainingIterations = numTrainingImages
    print "---TM training---"
    with open('logs.txt', 'w') as f:
        f.write('-----TM Training-----\n')
    start = time()
    for i in range(nTrainingIterations):
        if i % 200 == 199:
            t1 = time()
            mins = (t1 - start) / 60
            if mins >= 1:
                mins, seconds = math.modf(mins)
            else:
                mins, seconds = 0, mins
            mins = int(mins)
            seconds = int(seconds * 60)
            print "\t%d-th iteration,\t%.2d min %.2d sec" % (i, mins, seconds)
        if i % 100 < 3:
            log = True
        else:
            log = False
        net.run(imgIterations, log)

    '''
    # ----------------------------------------TM TRAINING----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 1)
    tm.setParameter("inferenceMode", 0)
    sp2.setParameter("learningMode", 0)
    sp2.setParameter("inferenceMode", 0)

    nTrainingIterations = numTrainingImages
    print "---TM training---"
    start = time()
    for i in range(nTrainingIterations):
        net.run(imgIterations, False)
    print 'Finished in %06.2f sec' % (time() - start)

    # ----------------------------------------SP2 TRAINING----------------------------------------
    classifier.setParameter("inferenceMode", 0)
    classifier.setParameter("learningMode", 0)
    sp.setParameter("learningMode", 0)
    sp.setParameter("inferenceMode", 1)
    tm.setParameter("learningMode", 0)
    tm.setParameter("inferenceMode", 1)
    sp2.setParameter("learningMode", 1)
    sp2.setParameter("inferenceMode", 0)

    nTrainingIterations = numTrainingImages
    print "---SP2 training---"
    start = time()
    for i in range(nTrainingIterations):
        net.run(imgIterations, False)
    print 'Finished in %06.2f sec' % (time() - start)

    # ----------------------------------------Classifier TRAINING----------------------------------------
    classifier.setParameter("inferenceMode", 1)
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
    print 'Finished in %06.2f sec' % (time() - start)


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
    numCorrect = 0
    for i in range(numTestImages):
        net.run(imgIterations, False)
        inferredCategory = classifier.getOutputData("categoriesOut").argmax()
        if sensor.getOutputData("categoryOut") == inferredCategory:
            numCorrect += 1
        if i % 60 == 59:
            print "\t%d-th iteration, nCorrect=%d" % (i, numCorrect)

    return (100.0 * numCorrect) / numTestImages
