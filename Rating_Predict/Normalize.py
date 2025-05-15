import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def f(col1, col2, df):
    cols = df.iloc[:,[i+1,j+1]].dropna()
    arr = cols.to_numpy()
    col1 = arr[:, 0]
    col2 = arr[:, 1]
    return np.dot(col1,col2)/(np.sqrt(np.dot(col1,col1)))/(np.sqrt(np.dot(col2,col2)))

df = pd.read_csv('MovieRecommender.csv')


subset = df.iloc[:, 1:]
subset = subset.mask(subset < 3, -1)
subset = subset.mask((subset >= 3), 1)
df.iloc[:, 1:] = subset
print(df.iloc[:,1:])


cor = np.empty((df.shape[1] - 1, df.shape[1] - 1))

for i in range(len(cor)):
    for j in range(len(cor)):
        if i == j:
            cor[i, j] = 1
        else:
            cor[i, j] = f(i,j,df)

sns.heatmap(cor,annot=False)
plt.show()

X = df.iloc[:,1:].to_numpy()

for user in range(X.shape[0]):
    has_rating = ~np.isnan(X[user])
    for movie in range(X.shape[1]):
        if not has_rating[movie]:
            X[user, movie] = np.dot(X[user, has_rating], cor[movie, has_rating]) / np.sum(np.abs(cor[movie, has_rating]))
    X[user, has_rating] = None

df.iloc[:,1:] = X

df.round(1).to_csv("Normalized_Recommendations.csv", index=False)