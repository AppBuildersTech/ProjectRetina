import configparser as cp
import json

config = cp.RawConfigParser()
config.read('config.cfg')

#  ------           [PATHS]          ------------- #
trndataPath = config.get("paths", "trndataPath")
tstdataPath = config.get("paths", "tstdataPath")
labelPath = config.get("paths", "labelPath")
classificationLabelPath = config.get("paths", "classificationLabelPath")
detectionLabelPath = config.get("paths", "detectionLabelPath")
samplePath = config.get("paths", "samplePath")
classificationFeatures = config.get("paths", "classificationFeatures")
detectionFeatures = config.get("paths", "detectionFeatures")
