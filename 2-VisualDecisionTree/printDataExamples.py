from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names) #input variables
print(iris.target_names) #output variables
print(iris.data[0]) #First row of data
print(iris.target[0]) #Output of first row
