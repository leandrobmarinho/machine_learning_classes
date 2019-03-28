from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data
y = iris.target

clf = Perceptron(tol=1e-3)
clf.fit(X, y)

print(clf.coef_)
print(clf.intercept_)
print(clf.predict(X))