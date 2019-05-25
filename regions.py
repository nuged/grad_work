from nupic.vision.regions.ImageSensorExplorers.BaseExplorer import BaseExplorer
from nupic.regions.sp_region import SPRegion
from nupic.regions.sdr_classifier_region import SDRClassifierRegion
from nupic.regions.knn_classifier_region import  KNNClassifierRegion
from nupic.vision.regions.ImageSensor import ImageSensor
import os, numpy, copy


class mySensor(ImageSensor):
    pass



class myExplorer(BaseExplorer):

    def __init__(self, updater, *args, **kwargs):
        BaseExplorer.__init__(self, *args, **kwargs)
        self.updater = updater

    def next(self, seeking=False):
        self.position = self.updater.getNextPosition(self.position)
        if self.position['image'] == self.numImages:
            self.position['image'] = 0

    def getImageIterations(self):
        return self.updater.getImageIterations()

    def first(self, center=True):
        BaseExplorer.first(self, False)
        self.position['offset'] = copy.deepcopy(self.updater.start)


class mySP(SPRegion):
    pass


class myClassifier(SDRClassifierRegion):
    pass


class secondExplorer(BaseExplorer):
    def __init__(self, start, length, step, *args, **kwargs):
        BaseExplorer.__init__(self, *args, **kwargs)
        self.start = start
        self.currentMove = 0
        self.length = length
        self.step = step
        self.firstIter = True

    def setMoveList(self, moveList):
        self.moveList = moveList

    def addAction(self, action):
        self.moveList.append(action)

    def next(self, seeking=False):
        if self.firstIter:
            self.firstIter = False
            return

        if self.currentMove == self.length - 1:
            self.currentMove = 0
            n = self.numImages // 10
            if self.position['image'] < self.numImages - n:
                self.position['image'] += n
            else:
                self.position['image'] -= self.numImages - n - 1
            self.position['offset'] = copy.deepcopy(self.start)
            if self.position['image'] == self.numImages:
                self.position['image'] = 0
            return

        move = self.moveList[self.currentMove]
        if move == 2:
            self.position['offset'][1] -= self.step
        elif move == 3:
            self.position['offset'][1] += self.step
        elif move == 0:
            self.position['offset'][0] -= self.step
        elif move == 1:
            self.position['offset'][0] += self.step

        self.currentMove += 1


    def first(self, center=True):
        BaseExplorer.first(self, False)
        self.position['offset'] = copy.deepcopy(self.start)
        self.firstIter = True
        self.currentMove = 0

    def getImageIterations(self):
        return self.length


