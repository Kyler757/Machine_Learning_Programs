import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

df = pd.read_csv("Shopping.csv")
df = df[["Age", "Gender", "Annual Income", "Spending Score"]]
df["Gender"] = df["Gender"].map({'Male' : 1, 'Female' : 0})
X = df.copy().to_numpy() * 1.0
std = np.std(X, axis=0)
av = np.average(X, axis=0)
X -= av
X /= std

kmeans = KMeans(n_clusters=16, n_init=30).fit(X)
centroids = kmeans.cluster_centers_ * std + av
cent = pd.DataFrame(centroids, columns=["Age", "Gender", "Annual Income", "Spending Score"]).round(2)
print(cent.to_string(index=False))


# plot the centroids
x = cent["Annual Income"]
y = cent["Spending Score"]
ages = cent["Age"]
genders = cent["Gender"]
colors = ['pink' if gender == 0 else 'blue' for gender in genders]

plt.figure(figsize=(10, 6))
for i in range(len(cent)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, edgecolors='black', zorder=2)

    text = plt.text(
        x[i], y[i], str(int(ages[i])),
        color='white', ha='center', va='center', fontsize=9, zorder=3
    )
    text.set_path_effects([
        path_effects.Stroke(linewidth=1.5, foreground='black'),
        path_effects.Normal()
    ])

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Cluster Centroids by Annual Income and Spending Score")
plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
plt.tight_layout()
plt.show()