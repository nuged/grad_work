from math import ceil


class BasePosUpdater:

    def __init__(self, imageSize, step, direction, n_imgs):
        self.imageSize = imageSize
        self.step = step
        self.n_imgs = n_imgs
        if direction == 'horizontal':
            self.idx = 1
        elif direction == 'vertical':
            self.idx = 0
        else:
            raise AttributeError()

    def getNextPosition(self, position):
        raise NotImplementedError()

    def getImageIterations(self):
        return int(ceil(self.imageSize / float(self.step))) ** 2

    def getNextImage(self, position):
        position['offset'] = [0, 0]
        position['image'] += 1
        return position
        N_IMGS = self.n_imgs
        if position['image'] + N_IMGS // 10 > N_IMGS - 1:
            position['image'] -= N_IMGS - 1 - N_IMGS // 10
        else:
            position['image'] += N_IMGS // 10
        position['offset'] = [0, 0]
        return position


class DiscPosUpdater(BasePosUpdater):

    def getNextPosition(self, position):
        offset = position['offset']
        idx = self.idx
        if offset[idx] < self.imageSize - self.step:
            position['offset'][idx] += self.step
        elif offset[1 - idx] < self.imageSize - self.step:
            position['offset'][1 - idx] += self.step
            position['offset'][idx] = 0
        else:
            position = self.getNextImage(position)
        return position


class ContPosUpdater(BasePosUpdater):

    def __init__(self, *args, **kwargs):
        BasePosUpdater.__init__(self, *args, **kwargs)
        self.straight = True

    def getNextPosition(self, position):
        offset = position['offset']
        idx = self.idx
        if self.straight:
            if offset[idx] < self.imageSize - self.step:
                position['offset'][idx] += self.step
            elif offset[1 - idx] < self.imageSize - self.step:
                position['offset'][1 - idx] += self.step
                self.straight = False
            else:
                position = self.getNextImage(position)
                self.straight = True
        else:
            if offset[idx] >= self.step:
                position['offset'][idx] -= self.step
            elif offset[1 - idx] < self.imageSize - self.step:
                position['offset'][1 - idx] += self.step
                self.straight = True
            else:
                position = self.getNextImage(position)
                self.straight = True
        return position


class specificUpdater:
    def __init__(self, step, moveList, start=[5, 5]):
        self.start = start
        self.step = step
        self.moveList = moveList
        self.currentMove = 0

    def getNextPosition(self, position):
        move = self.moveList[self.currentMove]
        if move == 2:
            position['offset'][1] -= self.step
        elif move == 3:
            position['offset'][1] += self.step
        elif move == 0:
            position['offset'][0] -= self.step
        elif move == 1:
            position['offset'][0] += self.step

        if self.currentMove == len(self.moveList) - 1:
            self.currentMove = 0
            position['image'] += 1
            position['offset'] = self.start
        else:
            self.currentMove += 1
        return position

    def getImageIterations(self):
        return len(self.moveList)
