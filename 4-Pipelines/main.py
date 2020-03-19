from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

#split dataset into trainer set and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

"""
#Use tree to train
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
"""

#Use Kneighbors instead of tree classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
#There are many different kinds of classifiers but they all are set up similarly to this

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

#get accuracy of predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
