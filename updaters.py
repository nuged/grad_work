from math import ceil


class BasePosUpdater:

    def __init__(self, imageSize, step, direction):
        self.imageSize = imageSize
        self.step = step
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
            position['image'] += 1
            position['offset'] = [0, 0]
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
                position['image'] += 1
                position['offset'] = [0, 0]
                self.straight = True
        else:
            if offset[idx] >= self.step:
                position['offset'][idx] -= self.step
            elif offset[1 - idx] < self.imageSize - self.step:
                position['offset'][1 - idx] += self.step
                self.straight = True
            else:
                position['image'] += 1
                position['offset'] = [0, 0]
                self.straight = True
        return position
