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

def writeResults(filename, data, n_patterns, result, time, first=False):
    stoplist = ['spatialImp', 'spVerbosity', 'seed', 'verbosity', 'inputWidth', 'temporalImp', 'globalInhibition',
                'checkSynapseConsistency']

    data['numPatterns'] = n_patterns
    data['result'] = result
    data['time'] = time

    csv_file = filename
    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[key for key in data if key not in stoplist])
            if first:
                writer.writeheader()
            writer.writerow({key: data[key] for key in data if key not in stoplist})
    except IOError:
        print("I/O error")
    data.pop('time')
    data.pop('result')
    data.pop('numPatterns')


def nextParamPos(positions, max):
    for i in range(len(positions)-1, -1, -1):
        if positions[i] < max[i] - 1:
            positions[i] += 1
            break
        else:
            positions[i] = 0
    return positions


def genParams(baseParams, source):
    n_params = len(source)
    current_positions = [0 for param in source]
    max_positions = [len(source[param]) for param in source]
    n_iters = np.prod(max_positions)
    for i in range(n_iters):
        current_params = {param : source[param][current_positions[j]] for j, param in enumerate(source)}
        baseParams.update(current_params)
        yield baseParams
        current_positions = nextParamPos(current_positions, max_positions)


if __name__ == '__main__':
    print getOffset()