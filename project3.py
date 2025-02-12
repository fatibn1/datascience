import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import joblib
import numpy as np

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
    ]
)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Baseline Model for Comparison (Mean Prediction)
baseline_prediction = np.mean(y_train)
baseline_mae = mean_absolute_error(y_test, [baseline_prediction] * len(y_test))

# XGBoost Model with GridSearchCV
xgb_model = XGBRegressor()
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='neg_mean_absolute_error', verbose=1)
xgb_grid.fit(X_train, y_train)

# Evaluate XGBoost
xgb_pred = xgb_grid.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# LightGBM Model with GridSearchCV
lgbm_model = LGBMRegressor()
lgbm_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

lgbm_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, scoring='neg_mean_absolute_error', verbose=1)
lgbm_grid.fit(X_train, y_train)

# Evaluate LightGBM
lgbm_pred = lgbm_grid.predict(X_test)
lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
lgbm_rmse = mean_squared_error(y_test, lgbm_pred)
lgbm_r2 = r2_score(y_test, lgbm_pred)

# Compare Models & Save Best
if xgb_mae < lgbm_mae:
    best_model = xgb_grid.best_estimator_
    model_name = "XGBoost"
else:
    best_model = lgbm_grid.best_estimator_
    model_name = "LightGBM"

joblib.dump(best_model, 'delivery_time_model.pkl')

# Print Model Evaluation
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"XGBoost - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.2f}")
print(f"LightGBM - MAE: {lgbm_mae:.2f}, RMSE: {lgbm_rmse:.2f}, R²: {lgbm_r2:.2f}")
print(f"Best Model: {model_name}")

# Streamlit App
st.title('Glovo Delivery Time Prediction')

# Input fields
distance = st.number_input('Distance (km)', min_value=0.1, format="%.2f")
order_volume = st.number_input('Order Volume', min_value=1, format="%d")
hour = st.number_input('Hour of the Day', min_value=0, max_value=23, format="%d")
day = st.number_input('Day of the Month', min_value=1, max_value=31, format="%d")
traffic = st.selectbox('Traffic Conditions', ['low', 'medium', 'high'])
weather = st.selectbox('Weather Conditions', ['clear', 'rainy', 'stormy'])

# Button to Trigger Prediction
if st.button('Predict'):
    if distance > 0 and order_volume > 0 and 0 <= hour <= 23 and 1 <= day <= 31:
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
    else:
        st.warning("Please fill in all fields correctly before predicting.")