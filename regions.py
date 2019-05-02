from nupic.vision.regions.ImageSensorExplorers.BaseExplorer import BaseExplorer
from nupic.regions.sp_region import SPRegion
from nupic.regions.sdr_classifier_region import SDRClassifierRegion
from nupic.regions.knn_classifier_region import  KNNClassifierRegion
from nupic.vision.regions.ImageSensor import ImageSensor


class mySensor(ImageSensor):

    @classmethod
    def getSpec(cls):
        ns = ImageSensor.getSpec()

        ns['inputs']['feedback'] = {'count': 0,
                                    'dataType': 'Real32',
                                    'description': 'Feedback',
                                    'isDefaultInput': True,
                                    'regionLevel': False,
                                    'requireSplitterMap': False,
                                    'required': True}
        return ns


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
        return BaseExplorer.first(self, False)


class mySP(SPRegion):
    pass


class myClassifier(KNNClassifierRegion):
    pass
