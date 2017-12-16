# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import argparse
import itertools
import pandas as pd
from config import *
from matplotlib import pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, RFE
from scipy.spatial.distance import mahalanobis
from tabulate import tabulate

def pltConfusion(cm, classes, title, cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def outlierDetection(data,err = 0.3):
	mean = np.mean(data, axis=0)
	covMat = np.cov(np.transpose(data))
	outliers = list()

	for i in range(len(data)):
		dist = mahalanobis(data.iloc[i],mean,np.linalg.inv(covMat))
		if dist > err:
			outliers.append(i)

	return outliers

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset',type=str,default="all",help='Choose among dataset options.')
parser.add_argument('-m','--mode',type=str,default="classification",help='Choose between detection or classification.')
parser.add_argument('-o','--outlier',type=int,default=10,help='Mahalanobis distance for outlier detection')
parser.add_argument('-s','--selection',type=str,default="chi2",help='Choose a feature selection method: chi2, f_classif, mutual_info_classif')
parser.add_argument('-p','--split',type=int,default=5,help='Choose the number of splits you want to have for dataset.')
parser.add_argument('-f','--nfeat',type=int,default=5,help='Choose the number of features you want to use.')

args = parser.parse_args()

classifiers = [
    MLPClassifier(hidden_layer_sizes=(100,100),learning_rate_init=0.03),
    KNeighborsClassifier(n_neighbors=30),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    ExtraTreeClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    AdaBoostClassifier(),
	LinearDiscriminantAnalysis(),
	QuadraticDiscriminantAnalysis(),
	NearestCentroid()
]

names = [
    "MLP",
    "K-Nearest Neighbors",
    "Decision Tree(CART)",
    "Random Forest",
	"ExtraTreeClassifier",
    "Linear SVM",
    "RBF SVM",
    "AdaBoost",
    "LDA",
    "QDA",
	"Nearest Centroid",
]

# You can either choose to detect diabetic retinopathy from the images or classify them to five different severity levels.
if args.mode == "detection":
	df = pd.read_csv(detectionFeatures)
	df.loc[df.label > 1, 'label'] = 1
elif args.mode == "classification":
	df = pd.read_csv(classificationFeatures)

# Take data and target classes
data = df.iloc[:,1:-2]
trgt = df.iloc[:,-1]

# Since there are many outliers in the data we wanted to at least get rid of obvious ones. (Ex. 35766_right.jpeg)
outliers = outlierDetection(data,args.outlier)

# Simple normalization is applied. (std(data) * (max(data) - min(data)) + max(data))
data = MinMaxScaler().fit_transform(data)

# Feature selection applied here. There are a couple of options in scikit-learn: chi-squared stats, ANOVA F-value, mutual information estimation.
data = SelectKBest(eval(args.selection), k=args.nfeat).fit_transform(data, trgt)

# Decide how to split dataset.
kf = KFold(n_splits=args.split)

# Classify data according to given parameters.
kappaScore = dict(); confusionM = dict(); tabularData = list();
for c,(name,clf) in enumerate(zip(names,classifiers)):
	kappaScore[name] = 0
	confusionM[name] = 0

for trnidx, tstindx in kf.split(data):
	trndata, tstdata = data[trnidx], data[tstindx]
	trntrgt, tsttrgt = trgt[trnidx], trgt[tstindx]

	kappaList = list()
	for c,(name,clf) in enumerate(zip(names,classifiers)):
		clf.fit(trndata,trntrgt)
		ypred = clf.predict(tstdata)
		confusionM[name] += confusion_matrix(tsttrgt, ypred)
		kappaScore[name] += cohen_kappa_score(tsttrgt,ypred,weights="quadratic") / args.split

# Print out kappa scores for each classifier
for name in names:
	accuracy = np.trace(confusionM[name]) / float(confusionM[name].sum())
	tabularData.append([name,kappaScore[name],accuracy])
tabularData = np.vstack(tabularData)
print tabulate(tabularData, headers=["Classifier", "KappaScore", "Accuracy"],floatfmt=".3f")

# Plot confusion matrices for each classifier.
# for c,(name,clf) in enumerate(zip(names,classifiers)):
# 	plt.figure()
# 	pltConfusion(confusionM[name], classes=np.unique(trgt),title='Confusion matrix for ' + name + ' classifier ($\kappa$ = ' + "{0:.2f}".format(kappaScore[name]) + ").")
# 	plt.show()
