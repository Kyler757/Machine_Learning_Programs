import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("Shopping.csv")
df = df[["Age", "Gender", "Annual Income", "Spending Score"]]
df["Gender"] = df["Gender"].map({'Male' : 1, 'Female' : 0})
X = df.copy().to_numpy() * 1.0
std = np.std(X, axis=0)
av = np.average(X, axis=0)
X -= av
X /= std

start, end, points = 1, 40, 15
inertia = np.zeros(points)
clusters = np.linspace(start, end, points, dtype=np.uint)
for i in range(len(clusters)):
    print(clusters[i])
    kmeans = KMeans(n_clusters=clusters[i], n_init=30).fit(X)
    inertia[i] = kmeans.inertia_

plt.plot(clusters, inertia)
plt.show()