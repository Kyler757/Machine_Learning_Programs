import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")

df = df.query('state == "Minnesota"')

df.dropna(subset=['house_size'], inplace=True)
df.dropna(subset=['acre_lot'], inplace=True)
df.dropna(subset=['price'], inplace=True)

df['bed'] = df['bed'].fillna(0)
df['bed'] = df['bed'].astype(int)

df['bath'] = df['bath'].fillna(0)
df['bath'] = df['bath'].astype(int)

stats = df[['price', 'bed', 'bath', 'acre_lot', 'house_size']]

corr_matrix = stats.corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()