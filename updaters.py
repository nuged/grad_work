import copy

class specificUpdater:
    def __init__(self, step, moveList, start):
        self.start = start
        self.step = step
        self.moveList = moveList
        self.currentMove = 0

    def getNextPosition(self, position):
        #print position
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
            position['offset'] = copy.deepcopy(self.start)
        else:
            self.currentMove += 1
        return position

    def getImageIterations(self):
        return len(self.moveList)
