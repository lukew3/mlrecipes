#Pipeline is the same except for the classifier
#Function for finding distance
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)

#Classifier creation
import random
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if (dist < best_dist):
                best_dist = distance
                best_index = i
        return self.y_train[best_index]


#imoprt dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

#split dataset into trainer set and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

#get accuracy of predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
