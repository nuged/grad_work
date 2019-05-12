import numpy as np


def getIndex(offset, imgSize=28, step=7):
    offset[0] //= step
    offset[1] //= step
    return offset[0] * imgSize // step + offset[1]


def getOffset(idx, imgSize=28, step=7):
    offset = [0, 0]
    offset[1] = idx % (imgSize // step) * step
    offset[0] = idx // (imgSize // step) * step
    return offset


def updateState(state, action, imgSize=28, step=7):
    gridSize = imgSize // step
    offset = getOffset(state % gridSize**2)
    if action == 0:
        offset[0] -= step
    elif action == 1:
        offset[0] += step
    elif action == 2:
        offset[1] -= step
    elif action == 3:
        offset[1] += step
    state = getIndex(offset) + gridSize**2 * (action + 1)
    return state


def getPossibleActions(offset, imgSize=28, windowSize=10, step=7):
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


if __name__ == '__main__':
    print getOffset(215)
    a = getPossibleActions([7, 19])
    b = np.full(4,np.nan)

    b[a] = np.random.uniform(-100, 0, a.shape[0])
    print b