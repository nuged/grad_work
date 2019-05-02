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
