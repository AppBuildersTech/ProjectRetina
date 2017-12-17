import numpy as np
import os, sys
import pandas as pd
from config import *

sampleSize = 300
imlabels = pd.read_csv(labelPath)
labels = np.unique(imlabels['level'])
df = pd.DataFrame(columns=list(imlabels))
for l in labels:
	if l == 0:
		df = df.append(imlabels.loc[imlabels['level'] == l].sample(700))
	else:
		df = df.append(imlabels.loc[imlabels['level'] == l].sample(700))

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(classificationLabelPath)
