import numpy as np
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

import matplotlib.pyplot as plt
plt.plot(X,y,"b.")
X_poly.shape
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_poly_new = poly_features.fit_transform(X_new)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
y_poly_pred = lin_reg.predict(X_poly_new)

plt.plot(X,y,"b.")
plt.plot(X_new, y_poly_pred,"r-")