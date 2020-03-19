#identifying apples and oranges
from sklearn import tree
features = [[140, 1], [130, 1], [150, 0], [170, 0]] # where 1 is smooth and 0 is bumpy
labels = [0, 0, 1, 1] #Where 0 is apple and 1 is orange
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))
