import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics



data = pd.read_csv("C:\\Users\\user\PycharmProjects\\final_project\src\project\data\data.csv")
X = data.drop(labels=['diameter'], axis=1)
y = data['diameter']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=42)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

k = 10
knn_regressor = KNeighborsRegressor(n_neighbors = k).fit(X_train,y_train)

dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train,y_train)

svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train,y_train)

models = [linear_reg, rf_regressor, knn_regressor, dt_regressor, svr_regressor]
model_names = ['Linear Regression', 'Random Forest','KNN','Decision Tree','Support Vector Regression']
best_model = None
best_r2_score = 0


for model, name in zip(models, model_names):
    y_val_pred = model.predict(X_val)
    r2 = metrics.r2_score(y_val, y_val_pred)  # Use y_val instead of y_test
    print(f"Model: {name}, R2-Score: {r2:.4f}")

    if r2 > best_r2_score:
        best_r2_score = r2
        best_model = model

# Test the best model on the test set
y_test_pred = best_model.predict(X_test)
print("Best Model (on Test Set):", type(best_model).__name__)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
mape = np.mean(np.abs((y_test - y_test_pred) / np.abs(y_test)))
print('Mean Absolute Percentage Error (MAPE):', round(mape , 2))
print('R-Squared:', metrics.r2_score(y_test, y_test_pred))
print('Accuracy:', round((1 - mape), 2))