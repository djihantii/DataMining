from sklearn import *
from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


iris=datasets.load_iris()
data = iris.data
feature_names=iris.feature_names
#print (iris)
#print (iris.data)
#print (iris.target_names)
#print(iris.target)

#print (iris)
#print (iris.feature_names)
#print(iris.target_names)

#print (iris)
#print (iris.data)

#for i in range(len(iris.data)) :
   # print("la donnee ", iris.data[i], " de la classe ",iris.target_names[iris.target[i]])
    
#print(iris.data.mean(0))
#print(iris.data.mean(1))

#print(iris.target.size)

#print(iris.data.mean(0))
#print(data.min(0))
#print(data.max(0))

#print(data.std(0))

#print(data.size)

#print(data.shape)
#print(size(feature_names))
#print(size(iris.target_names))

if __name__ == "__main__":
    iris = sk.datasets.load_iris()
    print(iris)
    print("Number of data : ", len(iris.data))
    print("Names of variables : ", ' ,'.join(str(e) for e in iris.feature_names) )
    print("Names of classes : ", ' ,'.join(str(e) for e in iris.target_names) )
    print("Objects and their classes")
    print('\n'.join('object n°'+str(n)+"'s class is "+iris.target_names[iris.target[n]] for n,o in enumerate(iris.data)))

