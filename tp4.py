
# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import sklearn as sk
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.cluster import adjusted_rand_score
import math


def distance(d1, d2):
    return math.sqrt(np.sum((d1 - d2)**2))

def generateCentroid(data, ncluster):
    centroids = []
    for i in range(ncluster):
        c=random.choice(data)
        centroids.append(c)
    return centroids

def computeClass(data_dict, centroids):
    distances = []
    for c in centroids:
        distances.append(distance(data_dict,c))
    #return distances.index(min(distances))
    return np.array(distances).argmin()

def recompteCentroid(data_dict, classes, ncluster):
    centroids = []
    for i in range(ncluster):
        iclassdata_dict = data_dict[classes == i]
        if len(iclassdata_dict) > 0 :
            centroids.append(np.average(iclassdata_dict, axis=0))
        else:
            #c = np.random.rand(*(data_dict[0].shape))
            c = np.ones_like(data_dict[0])
            centroids.append(c)
    return centroids
    
    
    

def myKmeans(data, ncluster):
    centroids = generateCentroid(data, ncluster)
    classes = []
    for d in data:
        classes.append(computeClass(d, centroids))
    newCentroids=recompteCentroid(data, np.array(classes), ncluster)
    while(not np.array_equal(newCentroids, centroids)):
        centroids = newCentroids
        classes = []
        for d in data:
            classes.append(computeClass(d, centroids))
        newCentroids=recompteCentroid(data, np.array(classes),ncluster)
    return centroids, classes
        
        
    

ncluster = 5
iris = sk.datasets.load_iris()
myclasses = myKmeans(iris.data, ncluster)[1]
skclasses = sk.cluster.k_means(iris.data, ncluster)[1]
#print(myclasses)
#print(skclasses)
print(adjusted_rand_score(skclasses, myclasses))

def applySilhouette(clusteringMethod, data, range_n_clusters = range(2, 10), n_times = 10):
    general_sil = []
    for nbcluster in range_n_clusters:
        local_sil = []
        for repeat in range(n_times):
            classes = clusteringMethod(data, nbcluster)
            local_sil.append(sk.metrics.silhouette_score(data, classes))
        value = np.array(local_sil).mean()
        general_sil.append(value)
        print("For n_clusters =", nbcluster,
              "The average silhouette_score is :", value)
    return range_n_clusters[np.array(general_sil).argmax()]

best_n_cluster = applySilhouette(lambda data, n : myKmeans(data, n)[1], iris.data)
print("Le meilleur nombre de cluster : ", best_n_cluster)    
        
    
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

#Choix projet
import csv
       

with open('choixprojets.csv') as csvfile:
    choixprojetdata_dict = csv.reader(csvfile, delimiter = ";")
    C = []
    M = []
    mention_to_int = {
            "jamais" : 0,
            "moyen" : 1,
            "bien" : 2,
            "trèsbien" : 3 
        }
    etudiant_to_int = {}
    nb_etudiant = 0
    projet_to_int = {}
    nb_projet = 0
    data_dict = {}
    for i,row in enumerate(choixprojetdata_dict):
        if i>0:
            etudiant, projet, mention = row
            if etudiant not in data_dict:
                data_dict[etudiant] = {}
            data_dict[etudiant][projet] = mention
            if etudiant in etudiant_to_int:
                etudiant = etudiant_to_int[etudiant]
            else:
                etudiant_to_int[etudiant] = nb_etudiant
                etudiant = nb_etudiant
                nb_etudiant += 1
            
            if projet in projet_to_int:
                projet = projet_to_int[projet]
            else:
                projet_to_int[projet] = nb_projet
                projet = nb_projet
                nb_projet += 1
            mention = mention_to_int[mention]
            C.append(etudiant)
            M.append([projet, mention])
    
    compl = 0
    for etudiant in etudiant_to_int:
        for projet in projet_to_int: 
            if projet not in data_dict[etudiant]:
                compl += 1
                C.append(etudiant_to_int[etudiant])
                M.append([projet_to_int[projet], mention_to_int["moyen"]])
            
    C = np.array(C)
    M = np.array(M)
    data = np.insert(M, 0, C, axis = 1) 
    print("Donnee completee : ", compl)
    shil_values = []
    list_of_clustering_methods = [
            ("AgglomerativeClustering", sk.cluster.AgglomerativeClustering), 
            ("Kmeans", sk.cluster.KMeans),
            ]
    for name, clustering_met in list_of_clustering_methods:
        classifier = clustering_met(n_clusters = nb_projet)
        classes = classifier.fit_predict(data)
        shil_value = sk.metrics.silhouette_score(data, classes)
        shil_values.append(shil_value)
        print("Silhouette value for {} is {}".format(name, shil_value))
    best_meth_idx = np.array(shil_values).argmax()
    print("The best method is : {}".format(list_of_clustering_methods[best_meth_idx][0]))
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        