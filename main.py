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

    net = createNetwork(baseParameters)

    train(net, 'mnist')

    updater = getUpdater(mode, step, direction, 1000)

    net.regions['sensor'].setParameter('explorer', yaml.dump(["regions.myExplorer", {"updater": updater}]))

    pctCorrect = test(net, 'mnist')
    print 'Pct of correct: ', pctCorrect

    currentResults = {}

    currentResults['step'] = int(step)
    currentResults['mode'] = mode
    currentResults['direction'] = direction
    currentResults['windowSize'] = int(windowSize)
    currentResults['pctCorrect'] = pctCorrect

    results = results.append(currentResults, ignore_index=True)
    results.to_csv('results.csv', index=False)
