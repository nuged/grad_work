import numpy as np
import csv

def getIndex(offset, gridSize, step):
    offset[0] //= step
    offset[1] //= step
    return offset[0] * gridSize + offset[1]


def getOffset(idx, gridSize, step):
    offset = [0, 0]
    offset[1] = idx % gridSize * step
    offset[0] = idx // gridSize * step
    return offset


def updateState(state, action, history, gridSize, step):
    # history contains at least 'action'
    offset = getOffset(state % gridSize**2, gridSize, step)
    if action == 0:
        offset[0] -= step
    elif action == 1:
        offset[0] += step
    elif action == 2:
        offset[1] -= step
    elif action == 3:
        offset[1] += step
    state = getIndex(offset, gridSize, step)    # idx in the '0th' matrix
    for i, act in enumerate(reversed(history)):
        state += (act + 1) * gridSize**2 * 4**i
    return state


def getPossibleActions(offset, imgSize, step):
    leftResult = offset[0] - step
    rightResult = offset[0] + step
    upResult = offset[1] - step
    downResult = offset[1] + step
    actions = []
    if leftResult >= 0:
        actions.append(0)
    if rightResult < imgSize:
        actions.append(1)
    if upResult >= 0:
        actions.append(2)
    if downResult < imgSize:
        actions.append(3)
    return np.array(actions)

def writeResults(data, filename, first=False):
    fields = ['columnCount', 'activationThreshold', 'cellsPerColumn', 'connectedPerm', 'permanenceInc',
              'minThreshold', 'permanenceDec', 'initialPerm', 'seed', 'newSynapseCount', 'predictedSegmentDecrement',
              'time', 'result']

    csv_file = filename
    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            if first:
                writer.writeheader()
            writer.writerow({key: data[key] for key in data if key in fields})
    except IOError:
        print("I/O error")
    data.pop('time')
    data.pop('result')

if __name__ == '__main__':
    print getIndex([6, 7], 4, 7)