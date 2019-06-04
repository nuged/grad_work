from nupic.vision.regions.ImageSensorExplorers.BaseExplorer import BaseExplorer
from nupic.regions.sp_region import SPRegion
from nupic.regions.sdr_classifier_region import SDRClassifierRegion
from nupic.regions.knn_classifier_region import  KNNClassifierRegion
from nupic.vision.regions.ImageSensor import ImageSensor
import os, numpy, copy, torch, yaml
from mlp import Net, AutoEncoder
from time import time
from nupic.bindings.regions.PyRegion import PyRegion
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np

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

class AERegion(PyRegion):
    @classmethod
    def getSpec(cls):
        """
        Overrides :meth:`nupic.bindings.regions.PyRegion.PyRegion.getSpec`.
        """
        ns = dict(
            description='Neural autoencoder in PyTorch',
            singleNodeOnly=True,

            inputs=dict(
                bottomUpIn=dict(
                    description='Belief values over children\'s groups',
                    dataType='Real32',
                    count=0,
                    required=True,
                    regionLevel=False,
                    isDefaultInput=True,
                    requireSplitterMap=False),
            ),

            outputs=dict(
                bottomUpOut=dict(
                    description='Hidden representation',
                    dataType='Real32',
                    count=0,
                    regionLevel=True,
                    isDefaultOutput=False,
                    requireSplitterMap=False),
            ),

            parameters=dict(
                inputWidth=dict(
                    description='Size of inputs to the autoencoder.',
                    accessMode='Read',
                    dataType='UInt32',
                    count=1,
                    constraints=''),

                learningMode=dict(
                    description='Boolean (0/1) indicating whether or not a region '
                                'is in learning mode.',
                    dataType='UInt32',
                    count=1,
                    constraints='bool',
                    defaultValue=1,
                    accessMode='ReadWrite'),

                inferenceMode=dict(
                    description='Boolean (0/1) indicating whether or not a region '
                                'is in inference mode.',
                    dataType='UInt32',
                    count=1,
                    constraints='bool',
                    defaultValue=1,
                    accessMode='ReadWrite'),

                hiddenSize=dict(
                    description='',
                    dataType='UInt32',
                    required=True,
                    count=1,
                    constraints='',
                    defaultValue=1024,
                    accessMode='Create'),

                learningRate=dict(
                    description='The learningRate is the learning rate of the classifier.'
                                'lower learningRate results in longer term memory and slower '
                                'learning',
                    dataType="Real32",
                    count=1,
                    constraints='',
                    defaultValue=0.001,
                    accessMode='Create'),

                momentum=dict(
                    description='',
                    dataType="Real32",
                    count=1,
                    constraints='',
                    defaultValue=0.9,
                    accessMode='Create'),

                optimizer=dict(
                    description='The classifier implementation to use.',
                    accessMode='ReadWrite',
                    dataType='Byte',
                    count=0,
                    constraints='enum: sgd, adam'),

                storageSize=dict(
                    description='',
                    dataType='UInt32',
                    required=True,
                    count=1,
                    constraints='',
                    defaultValue=7,
                    accessMode='Create'),

            ),
        )

        return ns

    def __init__(self,
                 inputWidth,
                 hiddenSize,
                 learningRate,
                 momentum,
                 optimizer,
                 storageSize,
                 **kwargs):
        self.learningMode = True
        self.inferenceMode = True

        self.inputWidth = inputWidth
        self.hiddenSize = hiddenSize

        self.storageSize = storageSize
        self.numStored = 0
        self.X = torch.zeros([storageSize, inputWidth], device='cuda')

        PyRegion.__init__(self, **kwargs)

        self.ae = AutoEncoder(inputWidth, hiddenSize)
        self.criterion = nn.BCELoss()
        self.ae.cuda()

        if optimizer == 'sgd':
            self.opt = optim.SGD(self.ae.parameters(), lr=learningRate, momentum=momentum)
        elif optimizer == 'adam':
            self.opt = optim.Adam(self.ae.parameters(), lr=learningRate)
        else:
            raise AttributeError

    def initialize(self):
        pass

    def setParameter(self, name, index, value):
        """
        Overrides :meth:`nupic.bindings.regions.PyRegion.PyRegion.setParameter`.
        """
        if name == "learningMode":
            self.learningMode = bool(int(value))
        elif name == "inferenceMode":
            self.inferenceMode = bool(int(value))
        else:
            return PyRegion.setParameter(self, name, index, value)

    def learn(self):
        if self.numStored == 0:
            return
        self.opt.zero_grad()
        outputData = self.ae(self.X)
        loss = self.criterion(outputData, self.X)
        loss.backward()
        self.opt.step()
        self.numStored = 0

    def compute(self, inputs, outputs):
        inputData = torch.as_tensor(inputs["bottomUpIn"]).view([1, -1]).cuda()

        if self.learningMode:
            self.X[self.numStored] = inputData
            self.numStored += 1

        with torch.no_grad():
            self.ae(inputData)
            outputs['bottomUpOut'][:] = self.ae.hidden.cpu().numpy()

        if self.numStored == self.storageSize:
            self.learn()

    def getOutputElementCount(self, name):
        """
        Overrides :meth:`nupic.bindings.regions.PyRegion.PyRegion.getOutputElementCount`.
        """
        if name == 'bottomUpOut':
            return self.hiddenSize
        else:
            raise Exception('Unknown output: ' + name)

class myClassifier(PyRegion):

    @classmethod
    def getSpec(cls):
        """
        Overrides :meth:`nupic.bindings.regions.PyRegion.PyRegion.getSpec`.
        """
        ns = dict(
            description='Neural Net in PyTorch',
            singleNodeOnly=True,

            inputs=dict(
                categoryIn=dict(
                    description='Vector of categories of the input sample',
                    dataType='Real32',
                    count=0,
                    required=True,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False),

                bottomUpIn=dict(
                    description='Belief values over children\'s groups',
                    dataType='Real32',
                    count=0,
                    required=True,
                    regionLevel=False,
                    isDefaultInput=True,
                    requireSplitterMap=False),
            ),

            outputs=dict(
                probabilities=dict(
                    description='Classification results',
                    dataType='Real32',
                    count=0,
                    regionLevel=True,
                    isDefaultOutput=False,
                    requireSplitterMap=False),
            ),

            parameters=dict(
                inputWidth=dict(
                    description='Size of inputs to the classifier.',
                    accessMode='Read',
                    dataType='UInt32',
                    count=1,
                    constraints=''),
                seqSize=dict(
                    description='',
                    accessMode='Read',
                    dataType='UInt32',
                    count=1,
                    constraints=''
                ),
                learningMode=dict(
                    description='Boolean (0/1) indicating whether or not a region '
                                'is in learning mode.',
                    dataType='UInt32',
                    count=1,
                    constraints='bool',
                    defaultValue=1,
                    accessMode='ReadWrite'),

                inferenceMode=dict(
                    description='Boolean (0/1) indicating whether or not a region '
                                'is in inference mode.',
                    dataType='UInt32',
                    count=1,
                    constraints='bool',
                    defaultValue=0,
                    accessMode='ReadWrite'),

                maxCategoryCount=dict(
                    description='The maximal number of categories the '
                                'classifier will distinguish between.',
                    dataType='UInt32',
                    required=True,
                    count=1,
                    constraints='',
                    # arbitrarily large value
                    defaultValue=2000,
                    accessMode='Create'),

                learningRate=dict(
                    description='The learningRate is the learning rate of the classifier.'
                                'lower learningRate results in longer term memory and slower '
                                'learning',
                    dataType="Real32",
                    count=1,
                    constraints='',
                    defaultValue=0.001,
                    accessMode='Create'),

                momentum=dict(
                    description='',
                    dataType="Real32",
                    count=1,
                    constraints='',
                    defaultValue=0.9,
                    accessMode='Create'),

                optimizer=dict(
                    description='The classifier implementation to use.',
                    accessMode='ReadWrite',
                    dataType='Byte',
                    count=0,
                    constraints='enum: sgd, adam'),

                storageSize=dict(
                    description='',
                    dataType='UInt32',
                    required=True,
                    count=1,
                    constraints='',
                    defaultValue=7,
                    accessMode='Create'),

            ),
        )

        return ns

    def __init__(self,
                 inputWidth,
                 learningRate,
                 optimizer,
                 momentum,
                 maxCategoryCount,
                 storageSize,
                 seqSize,
                 **kwargs):

        self.learningMode = True
        self.inferenceMode = False
        self.inputWidth = inputWidth
        self.maxCategoryCount = maxCategoryCount
        self.storageSize = storageSize
        self.seqSize = seqSize
        self.lossValues = []
        self.loss_idx = 0

        self.X = torch.zeros([storageSize, seqSize, inputWidth], device='cuda')
        self.y = torch.zeros([storageSize], dtype=torch.long, device='cuda')
        self.idx = 0
        self.numStored = 0

        PyRegion.__init__(self, **kwargs)

        self.net = Net(inputWidth * seqSize, maxCategoryCount)
        self.criterion = F.cross_entropy
        self.net.cuda()

        if optimizer == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=learningRate, momentum=momentum)
        elif optimizer == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=learningRate)
        else:
            raise AttributeError

    def initialize(self):
        pass

    def setParameter(self, name, index, value):
        """
        Overrides :meth:`nupic.bindings.regions.PyRegion.PyRegion.setParameter`.
        """
        if name == "learningMode":
            self.learningMode = bool(int(value))
        elif name == "inferenceMode":
            self.inferenceMode = bool(int(value))
        else:
            return PyRegion.setParameter(self, name, index, value)

    def learn(self):
        if self.numStored == 0:
            return

        self.opt.zero_grad()
        inputData = self.X.view([self.storageSize, -1])
        outputData = self.net(inputData)
        loss = self.criterion(outputData, self.y, reduction='none')
        loss = loss.mean()
        loss.backward()
        self.opt.step()
        self.numStored = 0
        self.X = torch.zeros_like(self.X)

    def compute(self, inputs, outputs):
        """
        Process one input sample.
        This method is called by the runtime engine.
        :param inputs: (dict) mapping region input names to numpy.array values
        :param outputs: (dict) mapping region output names to numpy.arrays that
               should be populated with output values by this method
        """

        inputData = torch.as_tensor(inputs["bottomUpIn"]).view([1, -1]).cuda()

        category = torch.as_tensor(inputs['categoryIn'], dtype=torch.long).cuda()

        self.X[self.numStored, self.idx] = inputData
        self.y[self.numStored] = category
        self.idx += 1

        with torch.no_grad():
            inputData = self.X[self.numStored].view([1, -1])
            outputData = self.net(inputData)
            outputs['probabilities'][:] = F.softmax(outputData, dim=1).cpu().numpy()
            if self.idx == self.seqSize:
                if self.loss_idx % 25 == 0:
                    loss = self.criterion(outputData, category, reduction='mean')
                    self.lossValues.append(loss.item())
                self.loss_idx += 1

        if self.idx == self.seqSize:
            self.idx = 0
            self.numStored += 1

        if self.numStored == self.storageSize:
            self.learn()

    def getOutputElementCount(self, name):
        """
        Overrides :meth:`nupic.bindings.regions.PyRegion.PyRegion.getOutputElementCount`.
        """
        if name == 'probabilities':
            return self.maxCategoryCount
        else:
            raise Exception('Unknown output: ' + name)


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
        pass

    def customNext(self):

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
