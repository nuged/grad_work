from nupic.vision.regions.ImageSensorExplorers.BaseExplorer import BaseExplorer
from nupic.regions.sp_region import SPRegion
from nupic.regions.sdr_classifier_region import SDRClassifierRegion
from nupic.regions.knn_classifier_region import  KNNClassifierRegion
from nupic.vision.regions.ImageSensor import ImageSensor
import os, numpy, copy

from nupic.bindings.math import GetNTAReal

_REAL_NUMPY_DTYPE = GetNTAReal()

class mySensor(ImageSensor):

    def Xcompute(self, inputs=None, outputs=None):
        """
        Generate the next sensor output and send it out.
        This method is called by the runtime engine.
        """
        # from dbgp.client import brk; brk(port=9019)
        if len(self._imageList) == 0:
            raise RuntimeError("ImageSensor can't run compute: no images loaded")

        # Check to see if new image belongs to a new sequence, if so force Reset
        prevPosition = self.prevPosition
        if prevPosition is not None:
            prevSequenceID = self._imageList[prevPosition["image"]]["sequenceIndex"]
        else:
            prevSequenceID = None

        self._updatePrevPosition()

        newPosition = self.prevPosition
        if newPosition is not None:
            newSequenceID = self._imageList[newPosition["image"]]["sequenceIndex"]
        else:
            newSequenceID = None

        if newSequenceID != prevSequenceID:
            self.prevPosition["reset"] = True

        # Get the image(s) to send out
        outputImages, finalOutput = self._getOutputImages()

        holdFor = self.explorer[2].holdFor
        self._holdForOffset += 1
        if self._holdForOffset >= holdFor:
            self._holdForOffset = 0  # pylint: disable=W0201
            self.explorer[2].next()
        self._iteration += 1

        # Compile information about this iteration and log it
        imageInfo = self._getImageInfo()
        if imageInfo["imagePath"] is None:
            filename = ""
        else:
            filename = os.path.split(imageInfo["imagePath"])[1]
        category = imageInfo["categoryIndex"]
        if category == -1:
            categoryName = ""
        else:
            categoryName = self.categoryInfo[category][0]
        self._logCommand([
            ("iteration", self._iteration),
            ("position", self.explorer[2].position),
            ("filename", filename),
            ("categoryIndex", category),
            ("categoryName", categoryName),
            ("erode", imageInfo["erode"]),
            ("blank", bool(self.prevPosition["reset"] and self.blankWithReset))
        ], None)

        # If we don"t have a partition ID at this point (e.g., because
        # of memory limits), then we need to try and pull from the
        # just-loaded image
        if imageInfo["partitionID"] is None:
            imgPosn = self.explorer[2].position["image"]
            imageInfo["partitionID"] = self._imageList[imgPosn].get("partitionID")

        if self.depth == 1:
            self.outputImage = outputImages[0]  # pylint: disable=W0201
        else:
            self.outputImage = outputImages  # pylint: disable=W0201

        # Invalidate the old location image
        self.locationImage = None  # pylint: disable=W0201

        # Log the images and locations if specified
        if self.logOutputImages:
            self._logOutputImages()
        if self.logOriginalImages:
            self._logOriginalImage()
        if self.logLocationImages:
            self._logLocationImage()

        # Save category to file
        self._writeCategoryToFile(category)

        if outputs:
            # Convert the output images to a numpy vector
            croppedArrays = [numpy.asarray(image.split()[0], _REAL_NUMPY_DTYPE)
                             for image in outputImages]
            # Pad the images to fit the full output size if necessary generating
            # a stack of images, each of them self.width X self.height
            pad = (self._cubeOutputs and
                   (self.depth > 1 or
                    croppedArrays[0].shape != (self.height, self.width)))
            if pad:
                fullArrays = [numpy.zeros((self.height, self.width), _REAL_NUMPY_DTYPE)
                              for i in xrange(self.depth)]
                for i in xrange(self.depth):
                    fullArrays[i][:croppedArrays[i].shape[0],
                    :croppedArrays[i].shape[1]] = croppedArrays[i]
            else:
                fullArrays = croppedArrays
            # Flatten and concatenate the arrays
            outputArray = numpy.concatenate([a.flat for a in fullArrays])

            # Send black and white images as binary (0, 1) instead of (0..255)
            if self.mode == "bw":
                outputArray /= 255
                outputArray = outputArray.round()

            # dataOut - main output
            if finalOutput is None:
                outputs["dataOut"][:] = outputArray
            else:
                outputs["dataOut"][:] = finalOutput

            # categoryOut - category index
            outputs["categoryOut"][:] = \
                numpy.array([float(category)], _REAL_NUMPY_DTYPE)

            # auxDataOut - auxiliary data
            auxDataOut = imageInfo["auxData"]
            if auxDataOut is not None:
                outputs["auxDataOut"][:] = auxDataOut

            # resetOut - reset flag
            if "resetOut" in outputs:
                outputs["resetOut"][:] = \
                    numpy.array([float(self.prevPosition["reset"])], _REAL_NUMPY_DTYPE)

            # bboxOut - bounding box
            if "bboxOut" in outputs and len(outputs["bboxOut"]) == 4:
                bbox = outputImages[0].split()[1].getbbox()
                if bbox is None:
                    bbox = (0, 0, 0, 0)
                outputs["bboxOut"][:] = numpy.array(bbox, _REAL_NUMPY_DTYPE)
                # Optionally log the bounding box information
                if self.logBoundingBox:
                    self._logBoundingBox(bbox)

            # alphaOut - alpha channel
            if "alphaOut" in outputs and len(outputs["alphaOut"]) > 1:
                alphaOut = numpy.asarray(outputImages[0].split()[1],
                                         _REAL_NUMPY_DTYPE).flatten()
                if not imageInfo["erode"]:
                    # Change the 0th element of the output to signal that the alpha
                    # channel should be dilated, not eroded
                    alphaOut[0] = -alphaOut[0] - 1
                outputs["alphaOut"][:alphaOut.shape[0]] = alphaOut

            # partitionOut - partition ID (defaults to zero)
            if "partitionOut" in outputs:
                partition = imageInfo.get("partitionID")
                if partition is None:
                    partition = 0
                outputs["partitionOut"][:] = numpy.array([float(partition)],
                                                         _REAL_NUMPY_DTYPE)


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


class myClassifier(KNNClassifierRegion):
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
            if self.position['image'] < 400:
                self.position['image'] += 100
            else:
                self.position['image'] -= 399
            self.position['offset'] = copy.deepcopy(self.start)
            if self.position['image'] == self.numImages - 1:
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


