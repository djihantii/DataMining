from sklearn import *
from sklearn.model_selection import *
import numpy as np
import matplotlib.pyplot as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from collections import Counter
from sklearn import svm


def KNN(target_names , X, Y, x_test, k):
    distances = metrics.pairwise.euclidean_distances(x_test, X)
    reordered = np.argsort(distances[0])
    vect = np.array([])
    for i in range(0, k):
        vect = np.append(vect, Y[reordered[i]])


    vote_labels = list(Counter(vect).keys())
    vote_count = list(Counter(vect).values())

    vote = np.argsort(vote_count)
    pred = vote_labels[vote[0]]
    # print('pred = ', int(pred))
    print('predicted class KNN= ', target_names[int(pred)])
    return pred

def KNN_errors(target_names, X, Y, k, number_test):
    vect_test = X[0:number_test+1]
    true = 0
    for i in range(0, number_test):
        pred=KNN(target_names , X, Y,[list(vect_test[i])], k)
        if(pred == Y[i]):
            true = true+1
    acc = float(true)/number_test*100
    print('accuracy KNN is = ', acc,'%')
    return acc

def KNN_sklearn(target_names, X, Y , test, k):
    clf = KNeighborsClassifier(n_neighbors = 5 )
    clf.fit(X , Y)
    Predicted_index = clf.predict(test)
    predicted_label = target_names[Predicted_index]
    print('predicted class KNN sklearn of data is ', predicted_label)
    return predicted_label

def KNN_Sklearn_errors(X, Y, k):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    predicted = knn.predict(x_test)
    acc = accuracy_score(y_test, predicted)*100
    print('accuracy = ', acc)
    return acc

def svm_method(X, Y):
    clf = svm.SVC()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    error = 1 -(accuracy_score(y_test, y_pred))
    print('pourcentage erreur SVM = ', error*100,'%')
    return error

def bayesian(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    error = 1 -(accuracy_score(y_test, y_pred))
    print('pourcentage erreur Bayesian = ', error*100,'%')
    return error

def decision_tree(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    estimator = DecisionTreeClassifier()
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    error = 1-(accuracy_score(y_test, y_pred))
    print('pourcentage erreur decision tree = ', error*100,'%')
    return error
if __name__ == '__main__':
    data = datasets.load_iris()
    x_data = data.data[:, :4]
    y_labels = data.target

    data_test = [[1.4, 3.6, 3.4, 1.2]]
    data_test2 = [[5.1 ,3.5 ,1.4 ,0.2]]
    data_test3 = [[5.9, 3.0, 5.0 , 3.8]]
    data_test4 = [[4.7, 3.9, 2.6, 3.5]]
    test = pd.DataFrame(data_test4)

    # KNN_errors(data.target_names , x_data, y_labels, 10, 80)
    # svm_method(data, data_test2)
    # KNN_sklearn(data.target_names,  x_data, y_labels, data_test2, 10)
    # KNN(data.target_names,  x_data, y_labels, data_test2, 10)

    svm_method(x_data, y_labels)
    bayesian(x_data, y_labels)
    decision_tree(x_data, y_labels)
