# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import sklearn as sk
from sklearn import datasets, decomposition, discriminant_analysis,  cluster, metrics

#from sklearn.lda import LDA
import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as cm
from itertools import combinations
import math

if __name__ == "__main__":
    iris = sk.datasets.load_iris()
    print(iris)
    print("Number of data : ", len(iris.data))
    print("Names of variables : ", ' ,'.join(str(e) for e in iris.feature_names) )
    print("Names of classes : ", ' ,'.join(str(e) for e in iris.target_names) )
    print("Objects and their classes")
    print('\n'.join('object n°'+str(n)+"'s class is "+iris.target_names[iris.target[n]] for n,o in enumerate(iris.data)))
    
    X = np.matrix([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
    #X = np.matrix([[1, 0, -1, 0]])
    print("Printing X")
    print(X)
    print("Printing the mean of X : ", X.mean());
    print("Printing the variance of X : ", X.var());
    
    sX = sk.preprocessing.scale(X)
    print("Printing scaled X")
    print(sX)
    print("Printing the mean of scaled X : ", sX.mean());
    print("Printing the variance of scaled X : ", sX.var());
    
    X2 = np.matrix([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
    print("Printing X2")
    print(X2)
    print("Printing the mean of the variables : ", X.mean(0));
    print("Printing the variance of the variables : ", X.var(0));
    
    MMscaler = sk.preprocessing.MinMaxScaler()
    MMscaler.fit(X2)
    sX2 = MMscaler.transform(X2)
    print("Printing scaled X2")
    print(sX2)
    print("Printing the mean of scaled X2 : ", sX2.mean());
    print("Printing the variance of scaled X2 : ", sX2.var());
    
    fig = pplot.figure()
    comb = list(combinations(range(len(iris.feature_names)), 2))
    nline = math.ceil(len(comb)/2)
    loAx = list()
    for n,(i,j) in enumerate(comb):
        loAx.append(fig.add_subplot(nline*100+20+n+1))
        loAx[-1].scatter(iris.data[:, i], iris.data[:, j], c=iris.target)
        loAx[-1].set_title(iris.feature_names[j]+' x '+iris.feature_names[i])
        loAx[-1].set_xlabel(iris.feature_names[i])
        loAx[-1].set_ylabel(iris.feature_names[j] )
    
    pplot.subplots_adjust(hspace=0.2)
    
    pca = sk.decomposition.PCA(n_components=best_n_cluster)
irisPCA = pca.fit_transform(iris.data)

lda = sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components=best_n_cluster)
irisLDA = lda.fit_transform(iris.data, iris.target)

fig = plt.figure()
pcaplot = fig.add_subplot(211)
pcaplot.scatter(irisPCA[:, 0], irisPCA[:, 1], c=iris.target)
pcaplot.set_title("Using PCA")
ldaplot = fig.add_subplot(212)
ldaplot.scatter(irisLDA[:, 0], irisLDA[:, 1], c=iris.target)
ldaplot.set_title("Using LDA")


