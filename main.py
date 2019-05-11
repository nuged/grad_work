from updaters import *
from network import createNetwork, train, test
import yaml
import pandas as pd
from time import time
from math import ceil
import gc, sys

IMG_SIZE = 28


def getUpdater(mode, step, direction, n_imgs):
    if mode == 'discrete':
        updater = DiscPosUpdater(IMG_SIZE, step, direction, n_imgs)
    elif mode == 'continuous':
        updater = ContPosUpdater(IMG_SIZE, step, direction, n_imgs)
    else:
        raise NotImplementedError()
    return updater


with open('parameters.yaml', "r") as f:
    baseParameters = yaml.safe_load(f)['modelParams']

tunableParameters = pd.read_csv('parameters.csv')

for i, row in tunableParameters.iterrows():
    mode = row['mode']
    step = int(row['step'])
    direction = row['direction']
    windowSize = row['windowSize']
    inputSize = windowSize * windowSize

    updater = specificUpdater(7, ['down', 'down', 'right', 'up', 'up', 'right'])

    baseParameters['SP']['inputWidth'] = inputSize
    baseParameters['sensor']['width'] = windowSize
    baseParameters['sensor']['height'] = windowSize
    baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

    #with open('test.csv', 'w') as f:
    #    f.write('nCols,syPermCon,potPct,time,result\n')

    #with open('logs.txt', 'w') as f:
    #    pass

    net = createNetwork(baseParameters)
    start = time()
    train(net, 'mnist')
    pctCorrect = test(net, 'mnist')
    t = time() - start

    print pctCorrect, '\ttook %f sec' % (t)

    exit(0)

    counter = 0
    for activationThreshold in [4, 8, 16]:
        for initialPerm in [0.1, 0.3, 0.6, 0.9]:
            for connectedPerm in [0.1, 0.3, 0.6, 0.9]:
                for permanenceInc in [0.01, 0.05, 0.1]:

                    counter += 1

                    newSynapseCount = int(ceil(1.5 * activationThreshold))
                    permanenceDec = permanenceInc / 10
                    predictedSegmentDecrement = permanenceInc * 0.05

                    claim = '\nstep %d:\n--------------actThr=%d, initPerm=%2.1f, conPerm=%2.1f, permInc=%3.2f-----------' \
                            % (counter, activationThreshold, initialPerm, connectedPerm, permanenceInc)

                    print claim

                    with open('logs.txt', 'a') as f:
                        print >>f, claim

                    baseParameters['TM']['activationThreshold'] = activationThreshold
                    baseParameters['TM']['initialPerm'] = initialPerm
                    baseParameters['TM']['connectedPerm'] = connectedPerm
                    baseParameters['TM']['permanenceInc'] = permanenceInc

                    baseParameters['TM']['permanenceDec'] = permanenceDec
                    baseParameters['TM']['newSynapseCount'] = newSynapseCount
                    baseParameters['TM']['predictedSegmentDecrement'] = predictedSegmentDecrement

                    net = createNetwork(baseParameters)
                    start = time()
                    train(net, 'mnist')
                    pctCorrect = test(net, 'mnist')
                    t = time() - start

                    with open('test.csv', 'a') as f:
                        answer = '%d,%2.1f,%2.1f,%3.2f,%6.3f,%4.2f\n' % \
                                 (activationThreshold, initialPerm, connectedPerm, permanenceInc, t, pctCorrect)
                        f.write(answer)

                    print pctCorrect, '\ttook %f sec' % (t)

    exit(0)



    counter = 0
    for numActiveColumnsPerInhArea in [40, 80, 160]:
        for synPermConnected in [0.1, 0.2, 0.4, 0.8]:
            for potentialPct in [0.1, 0.2, 0.4, 0.8]:

                counter += 1

                claim = '\n---------------step=%03d, nActCols=%d, synPerm=%2.1f, potPct=%2.1f-----------' \
                        % (counter, numActiveColumnsPerInhArea, synPermConnected, potentialPct)

                print claim

                baseParameters['SP2']['numActiveColumnsPerInhArea'] = numActiveColumnsPerInhArea
                baseParameters['SP2']['synPermConnected'] = synPermConnected
                baseParameters['SP2']['potentialPct'] = potentialPct

                net = createNetwork(baseParameters)
                start = time()
                train(net, 'mnist')
                pctCorrect = test(net, 'mnist')
                t = time() - start

                with open('test.csv', 'a') as f:
                    answer = '%03d,%2.1f,%2.1f,%5.2f,%4.2f\n' %\
                             (numActiveColumnsPerInhArea, synPermConnected, potentialPct, t, pctCorrect)
                    f.write(answer)

                print pctCorrect, '\ttook %f sec' % (t)
