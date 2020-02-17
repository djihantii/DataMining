from sklearn import *
import numpy as np
import matplotlib.pyplot as mp

def printCaracteristics(data):
    lenI = len(data.data)
    print('number of data = ',lenI )
    print('names of variables')
    for i in range(0, len(data.feature_names)):
        print(data.feature_names[i])

    print('names of classes')
    for j in range(0, len(data.target_names)):
        print(data.target_names[j])

    print('label and classes')
    for k in range(0, len(data.target)):
        print(data.target_names[data.target[k]])

def variables_average(data):
    return data.data.mean(0)
def data_average(data):
    return data.data.mean(1)

def variables_caracteristics(data):
    mean = variables_average(data)
    std = data.data.std(0)
    min = data.data.min()
    max = data.data.max()

    print('mean = ', mean)
    print('std = ',std)
    print('min = ',min)
    print('max = ',max)


if __name__ == '__main__':
    iris = datasets.load_iris()

    print(iris.data[0, :2])
    #printCaracteristics(iris)

    #variables_caracteristics(iris)
