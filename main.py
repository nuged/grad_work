from updaters import *
from network import createNetwork, train, test
import yaml
import pandas as pd
from time import time

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
results = pd.DataFrame(columns=tunableParameters.columns.union(['pctCorrect']))

for i, row in tunableParameters.iterrows():
    mode = row['mode']
    step = int(row['step'])
    direction = row['direction']
    windowSize = row['windowSize']
    inputSize = windowSize * windowSize

    updater = getUpdater(mode, step, direction, 10000)

    baseParameters['SP']['inputWidth'] = inputSize
    baseParameters['sensor']['width'] = windowSize
    baseParameters['sensor']['height'] = windowSize
    baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

    with open('test.csv', 'w') as f:
        f.write('SPCC,TMCC,TMCpC,SP2CC,time,result\n')

    for SPCC in [2048, 4096, 6144]:
        for TMCC in [2048, 4096, 6144]:
            for TMCpC in [4, 8, 16]:
                for SP2CC in [2048, 4096]:
                    baseParameters['SP']['columnCount'] = SPCC
                    baseParameters['SP']['numActiveColumnsPerInhArea'] = SPCC // 25
                    baseParameters['TM']['inputWidth'] = SPCC
                    baseParameters['TM']['columnCount'] = TMCC
                    baseParameters['TM']['cellsPerColumn'] = TMCpC
                    baseParameters['SP2']['columnCount'] = SP2CC
                    baseParameters['SP2']['numActiveColumnsPerInhArea'] = SP2CC // 25
                    baseParameters['SP2']['inputWidth'] = TMCC * TMCpC

                    print '\n----------SP_CC=%d, SP_AC=%d, TM_CC=%d, TM_CpC=%d, SP2_CC=%d, SP2_AC=%d----------' % \
                          (SPCC, SPCC // 25, TMCC, TMCpC, SP2CC, SP2CC // 25)

                    net = createNetwork(baseParameters)
                    start = time()
                    train(net, 'mnist')
                    pctCorrect = test(net, 'mnist')
                    t = time() - start

                    answer = '%d,%d,%d,%d,%f,%f\n' % (SPCC, TMCC, TMCpC, SP2CC, t, pctCorrect)

                    with open('test.csv', 'a') as f:
                        f.write(answer)

                    print pctCorrect, 'took %f sec' % t
