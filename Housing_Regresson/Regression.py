import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv")

df = df.query('state == "Minnesota"')

df.dropna(subset=['house_size'], inplace=True)
df.dropna(subset=['acre_lot'], inplace=True)
df.dropna(subset=['price'], inplace=True)

df['bed'] = df['bed'].fillna(0)
df['bed'] = df['bed'].astype(int)

df['bath'] = df['bath'].fillna(0)
df['bath'] = df['bath'].astype(int)

X = df[['bed', 'bath', 'house_size']]
y = df['price']

reg = LinearRegression().fit(X, y)
print("R^2", reg.score(X, y))

print("Root mean squared error", np.sqrt(np.average((y-reg.predict(X))**2.0)))
print("y-intercept", reg.intercept_)

coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': reg.coef_
})
print("\nCoefficients:")
print(coeff_df.to_string(index=False))