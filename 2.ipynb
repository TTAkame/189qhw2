import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# Load the dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['target'] = california.target

# Prepare the dataset for linear regression
X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit a linear regression model to the training data without regularization
LR = LinearRegression()
LR.fit(X_train, y_train)

# Evaluate the linear regression model on the test data
y_pred = LR.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Linear Regression:")
print("MSE: ", mse)
print("R-squared: ", r2)

# 4. Fit a ridge regression model to the training data with a regularization parameter of 𝜆 = 0.01
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, y_train)

# Evaluate the ridge regression model on the test data
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nRidge Regression:")
print("MSE: ", mse)
print("R-squared: ", r2)

# 5. Fit a Lasso regression model to the training data with a regularization parameter of 𝜆 = 0.01
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# Evaluate the lasso regression model on the test data
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nLasso Regression:")
print("MSE: ", mse)
print("R-squared: ", r2)

# 6. Fit an Elastic-Net regression model to the training data with a regularization parameter of 𝜆 = 0.01
eNet = ElasticNet(alpha=0.01)
eNet.fit(X_train, y_train)

# Evaluate the elastic-net regression model on the test data
y_pred = eNet.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nElastic-Net Regression:")
print("MSE: ", mse)
print("R-squared: ", r2)

# 7. Investigate the effect of different regularization parameter (𝜆) on coefficients of regression model

# Ridge regularization
lambdas_ridge = np.logspace(0, 4, 1000) # 1 to 10,000
coefs_ridge = []
for l in lambdas_ridge:
    ridge = Ridge(alpha=l, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs_ridge.append(ridge.coef_)

# Lasso regularization
lambdas_lasso = np.logspace(-5, 0, 1000) # 0.00001 to 1
coefs_lasso = []
for l in lambdas_lasso:
    lasso = Lasso(alpha=l, fit_intercept=False)
    lasso.fit(X_train, y_train)
    coefs_lasso.append(lasso.coef_)

# Elastic-Net regularization
lambdas_eNet = np.logspace(-5, 0, 1000) # 0.00001 to 1
coefs_eNet = []
for l in lambdas_eNet:
    eNet = ElasticNet(alpha=l, l1_ratio=0.5, fit_intercept=False)
    eNet.fit(X_train, y_train)
    coefs_eNet.append(eNet.coef_)

# Plot the coefficients vs. lambda for each regularization method
plt.figure(figsize=(18,6))

ax = plt.subplot(1,3,1)
ax.plot(lambdas_ridge, coefs_ridge)
ax.set_xscale('log')
plt.title('Ridge coefficients')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')

ax = plt.subplot(1,3,2)
ax.plot(lambdas_lasso, coefs_lasso)
ax.set_xscale('log')
plt.title('Lasso coefficients')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')

ax = plt.subplot(1,3,3)
ax.plot(lambdas_eNet, coefs_eNet)
ax.set_xscale('log')
plt.title('Elastic-Net coefficients')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')

plt.show()


