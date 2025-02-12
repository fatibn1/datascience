import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv('Large_Delivery_Time_Prediction_Dataset.csv')

# Feature Engineering
data['Order_Time'] = pd.to_datetime(data['Order_Time'])
data['Hour'] = data['Order_Time'].dt.hour
data['Day'] = data['Order_Time'].dt.day

# Define features and target
X = data.drop(columns=['Delivery_Time_Minutes', 'Order_Time'])
y = data['Delivery_Time_Minutes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical and categorical data
numerical_features = ['Distance_km', 'Order_Volume', 'Hour', 'Day']
categorical_features = ['Traffic_Conditions', 'Weather_Conditions']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

# XGBoost Model
xgb_model = XGBRegressor()
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='neg_mean_absolute_error')
xgb_grid.fit(X_train, y_train)

# LightGBM Model
lgbm_model = LGBMRegressor()
lgbm_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

lgbm_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, scoring='neg_mean_absolute_error')
lgbm_grid.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate XGBoost
xgb_pred = xgb_grid.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)
xgb_r2 = r2_score(y_test, xgb_pred)

# Evaluate LightGBM
lgbm_pred = lgbm_grid.predict(X_test)
lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
lgbm_rmse = mean_squared_error(y_test, lgbm_pred, squared=False)
lgbm_r2 = r2_score(y_test, lgbm_pred)

print(f"XGBoost - MAE: {xgb_mae}, RMSE: {xgb_rmse}, R²: {xgb_r2}")
print(f"LightGBM - MAE: {lgbm_mae}, RMSE: {lgbm_rmse}, R²: {lgbm_r2}")
import streamlit as st
import joblib

# Save the best model
best_model = xgb_grid.best_estimator_ if xgb_mae < lgbm_mae else lgbm_grid.best_estimator_
joblib.dump(best_model, 'delivery_time_model.pkl')

# Streamlit App
st.title('Glovo Delivery Time Prediction')

# Input fields
distance = st.number_input('Distance (km)')
order_volume = st.number_input('Order Volume')
hour = st.number_input('Hour of the Day')
day = st.number_input('Day of the Month')
traffic = st.selectbox('Traffic Conditions', ['low', 'medium', 'high'])
weather = st.selectbox('Weather Conditions', ['clear', 'rainy', 'stormy'])

# Preprocess input
input_data = pd.DataFrame({
    'Distance_km': [distance],
    'Order_Volume': [order_volume],
    'Hour': [hour],
    'Day': [day],
    'Traffic_Conditions': [traffic],
    'Weather_Conditions': [weather]
})

input_data = preprocessor.transform(input_data)

# Load model and predict
model = joblib.load('delivery_time_model.pkl')
prediction = model.predict(input_data)

st.write(f'Predicted Delivery Time: {prediction[0]:.2f} minutes')