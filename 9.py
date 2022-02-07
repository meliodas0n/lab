from sklearn.datasets import load_iris
iris = load_iris()
print("Features : ", iris.feature_names, 'Data : ', iris.data, 'Target : ', iris.target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.25)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

print("Accurcy : ", clf.score(x_test, y_test))

print("Predicted Data : ")
print(clf.predict(x_test))

predictions = clf.predict(x_test)

print("Test Data : ")
print(y_test)

diff = predictions - y_test
print("The Result is : ")
print(diff)

print("The number of samples missclassified : ", sum(abs(diff)))