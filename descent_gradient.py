from sklearn import *
from sklearn.model_selection import *
import numpy as np
import matplotlib.pyplot as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
e = 0.0001
nMax = 1000

def vizualise(x_values, y_values, x_label):
    plt.plot(x_values, y_values)
    label='x0 = '+str(x_label[0])+' n= '+str(x_label[1])
    plt.xlabel(label)
    plt.show()

def derivee(degree , equation):
    result = [0]*degree
    for i in range(0, degree):
        result[i] = equation[i+1]*(i+1)
    #
    # for j in range(0, degree):
    #     print(result[j])

    return result

def value_x(equation, x):
    result = 0
    for i in range(0, len(equation)):
        temp = np.power(x, i)
        result = result + equation[i]*temp

    # print('equation(x) = ', result)
    return result

def dg_iteration_max(x, equation, degree, n):

    for i in range(0, nMax):
        x = x - n*value_x(derivee(degree, equation), x)

    print('descent gradient = ', x)
    t = x - n*value_x(derivee(degree, equation), x)
    print('encore une iteration ', t)
    return x

def dg_e(x, equation, degree, n):
    iterations = 0
    x0 = x
    y = x - n*value_x(derivee(degree, equation), x)
    x_evolution = [x, y]
    y_evolution = [value_x(equation, x), value_x(equation, y)]
    while((x-y)>e):
        iterations = iterations+1
        x=y
        y = x - n*value_x(derivee(degree, equation), x)
        x_evolution = np.append(x_evolution, y)
        y_evolution = np.append(y_evolution, value_x(equation, y))

    print('x0 = ', x0, '   n = ', n)
    print('nombre d iteration = ', iterations)
    print('xmin = ', y)
    print('E(xmin) = ', value_x(equation, y))
    print('_________________________________________________________')
    # print('dg_e = ', y)
    vizualise(np.arange(iterations+2),x_evolution, [x0,n])
    return y

def compare_values(eq, degree, values, num_tests):
    results_x = [0]*num_tests
    results_y = [0]*num_tests
    for i in range(0, num_tests):
        r = dg_e(values[i][0], eq, degree, values[i][1])
        results_x[i] = r
        results_y[i] = value_x(eq, r)
    # for j in range(0, num_tests):
    #
    #     print('x0 = ',values[j][0], ' n= ',values[j][1] )
    #     print('x min = ', results_x[j])
    #     print('E(minLocal) = ', results_y[j])
    #     print('')
    #     print('******************************************************')


if __name__ == '__main__':

    eq = [30, -61, 41, -11, 1]
    equation = [2, 4, 3, 6]

    # xmin = dg_e(5, eq, 4, 0.17)
    # print('xmin = ', xmin)
    # print(value_x(eq, xmin))
    compare_values(eq, 4, [[5,0.001], [5,0.01],[5,0.1],[5,0.17],[5,1],[0,0.001]], 6)
