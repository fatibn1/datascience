import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st
import os

# Load the dataset
df = pd.read_csv('Large_Delivery_Time_Prediction_Dataset.csv')

# Feature Engineering
df['Order_Time'] = pd.to_datetime(df['Order_Time'])
df['Hour'] = df['Order_Time'].dt.hour
df['Day'] = df['Order_Time'].dt.day

# Define features and target
X = df.drop(columns=['Delivery_Time_Minutes', 'Order_Time'])
y = df['Delivery_Time_Minutes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical and categorical data
numerical_features = ['Distance_km', 'Order_Volume', 'Hour', 'Day']
categorical_features = ['Traffic_Conditions', 'Weather_Conditions']  # Ensure these column names match your dataset

# Check if column names are correct (case-sensitive)
print(X_train.columns)  # Print the actual column names in your DataFrame

# If there's a mismatch, correct the names in categorical_features
# For example, if the column is 'traffic_conditions':
# categorical_features = ['traffic_conditions', 'weather_conditions']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Handle unknown categories
    ])

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

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

# Evaluate XGBoost
xgb_pred = xgb_grid.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# Evaluate LightGBM
lgbm_pred = lgbm_grid.predict(X_test)
lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
lgbm_rmse = mean_squared_error(y_test, lgbm_pred)
lgbm_r2 = r2_score(y_test, lgbm_pred)

print(f"XGBoost - MAE: {xgb_mae}, RMSE: {xgb_rmse}, R²: {xgb_r2}")
print(f"LightGBM - MAE: {lgbm_mae}, RMSE: {lgbm_rmse}, R²: {lgbm_r2}")

# Save the best model
best_model = xgb_grid.best_estimator_ if xgb_mae < lgbm_mae else lgbm_grid.best_estimator_
joblib.dump(best_model, 'delivery_time_model.pkl')  # Save with the correct name

# Streamlit App
st.title('Delivery Time Prediction')

# Load Model
if os.path.exists('delivery_time_model.pkl'):
    model = joblib.load('delivery_time_model.pkl')
else:
    st.error("Model file not found. Please ensure the model is saved correctly.")
    st.stop()

# Input fields
distance = st.number_input('Distance (km)', min_value=0.1, max_value=50.0, step=0.1)
traffic = st.selectbox('Traffic Conditions', ['Low', 'Medium', 'High'])
weather = st.selectbox('Weather Conditions', ['Clear', 'Rainy', 'Snowy'])
order_volume = st.number_input('Order Volume', min_value=1, max_value=100, step=1)
hour = st.number_input('Hour of the Day', min_value=0, max_value=23, step=1)
day = st.number_input('Day of the Month', min_value=1, max_value=31, step=1)

# Preprocess input data
input_data = pd.DataFrame({
    'Distance_km': [distance],
    'Traffic_Conditions': [traffic],
    'Weather_Conditions': [weather],
    'Order_Volume': [order_volume],
    'Hour': [hour],
    'Day': [day]
})

# Apply preprocessing
input_data = preprocessor.transform(input_data)

# Predict
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Estimated Delivery Time: {prediction[4]:.2f} minutes')