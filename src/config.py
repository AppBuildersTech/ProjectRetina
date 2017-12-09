import configparser as cp
import json

config = cp.RawConfigParser()
config.read('config.cfg')

#  ------           [PATHS]          ------------- #
trndataPath = config.get("paths", "trndataPath")
tstdataPath = config.get("paths", "tstdataPath")
labelPath = config.get("paths", "labelPath")
sampleLabelPath = config.get("paths", "sampleLabelPath")
samplePath = config.get("paths", "samplePath")

# #  ------           [Compression]          ------------- #
frame_length = config.getint ("compression", "frame_length")
fs=config.getint ("compression", "fs")

#  ------           [FEATURES]          ------------- #
n_point = 1024
fft_length = 245760
