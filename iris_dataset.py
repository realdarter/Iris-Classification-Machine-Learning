from sklearn.datasets import load_iris

data = load_iris()
data.target[[10, 25, 50]]
print(list(data.target_names))
print(data)