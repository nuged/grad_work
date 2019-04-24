from updaters import *
from network import createNetwork, train, test
import yaml
import pandas as pd

IMG_SIZE = 28
CELLS_PER_COL = 4

baseParameters = {
    'TM' : {
        "cellsPerColumn": CELLS_PER_COL,
        'temporalImp' : 'cpp'
    },

    "SP" : {
        "spatialImp": "cpp",
        "synPermConnected": 0.2,
        "synPermActiveInc": 0.0,
        "synPermInactiveDec": 0.0,
        "globalInhibition": 1,
        "potentialPct": 0.9,
        "boostStrength": 0.0
    },

    "SP2" : {
        "spatialImp": "cpp",
        "synPermConnected": 0.2,
        "synPermActiveInc": 0.0,
        "synPermInactiveDec": 0.0,
        "globalInhibition": 1,
        "potentialPct": 0.9,
        "boostStrength": 0.0
    },

    "CLS" : {
        # "distThreshold": 0.000001,
        "maxCategoryCount": 10,
        #"k" : 25,
        # "cellsPerCol" : CELLS_PER_COL,
        #"distanceMethod": "pctOverlapOfInput"
        #"steps" : [1],
        "alpha" : 0.4,
        "verbosity" : 0,
        "implementation" : "cpp",
        "steps" : 1
    },

    "sensor" : {
        "mode": "bw",
        "background": 0
    }
}


def getUpdater(mode, step, direction):
    if mode == 'discrete':
        updater = DiscPosUpdater(IMG_SIZE, step, direction)
    elif mode == 'continuous':
        updater = ContPosUpdater(IMG_SIZE, step, direction)
    else:
        raise NotImplementedError()
    return updater


tunableParameters = pd.read_csv('parameters.csv')
results = pd.DataFrame(columns=tunableParameters.columns.union(['pctCorrect']))

for i, row in tunableParameters.iterrows():
    mode = row['mode']
    step = int(row['step'])
    direction = row['direction']
    windowSize = row['windowSize']
    columnCount = windowSize * 100
    activeColumns = windowSize * 5
    inputWidth = windowSize ** 2
    updater = getUpdater(mode, step, direction)


    baseParameters['TM']['columnCount'] = columnCount
    baseParameters['TM']['inputWidth'] = columnCount
    baseParameters['SP']['columnCount'] = columnCount
    baseParameters['SP']['inputWidth'] = inputWidth
    baseParameters['SP']['numActiveColumnsPerInhArea'] = activeColumns
    baseParameters['SP2']['columnCount'] = columnCount
    baseParameters['SP2']['inputWidth'] = columnCount * CELLS_PER_COL
    baseParameters['SP2']['numActiveColumnsPerInhArea'] = activeColumns
    baseParameters['sensor']['width'] = windowSize
    baseParameters['sensor']['height'] = windowSize
    baseParameters['sensor']['explorer'] = yaml.dump(["regions.myExplorer", {"updater": updater}])

    net = createNetwork(baseParameters)
    train(net, 'mnist')
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
