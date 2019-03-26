from scipy.spatial import distance
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
coords = [35.0456, -85.2672]

print(distance.euclidean(coords, coords))
print(distance.cityblock(coords, coords))
