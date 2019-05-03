from updaters import *
from network import createNetwork, train, test
import yaml
import pandas as pd

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

    with open('TM_test.csv', 'w') as f:
        f.write('cellsPerColumn,initialPerm,maxSegmentsPerCell,result\n')

    for cellsPerColumn in [2, 4, 8, 16]:
        for initialPerm in [0.05, 0.1, 0.3, 0.5, 0.7]:
            for maxSegmentsPerCell in [48, 64, 80, 96]:
                print '\n--------------------cellsPerCol=%d, initPerm=%f, maxSeg=%d--------------------' \
                      % (cellsPerColumn, initialPerm, maxSegmentsPerCell)
                baseParameters['TM']['cellsPerColumn'] = cellsPerColumn
                baseParameters['TM']['initialPerm'] = initialPerm
                baseParameters['TM']['maxSegmentsPerCell'] = maxSegmentsPerCell

                baseParameters['SP2']['inputWidth'] = baseParameters['TM']['columnCount'] * cellsPerColumn

                net = createNetwork(baseParameters)
                train(net, 'mnist')
                pctCorrect = test(net, 'mnist')

                with open('TM_test.csv', 'a') as f:
                    f.write(str(cellsPerColumn)+','+str(initialPerm)+','+str(maxSegmentsPerCell)+','+str(pctCorrect)+'\n')
