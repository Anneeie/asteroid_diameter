import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


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



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()

model.add(Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

loss = model.evaluate(X_test_scaled, y_test)
print('Test loss:', loss)

y_pred = model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R^2 Score: {r2}')
