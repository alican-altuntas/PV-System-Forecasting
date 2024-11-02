import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from datetime import timedelta
import matplotlib.pyplot as plt


def e2109957_altuntas_GF(input_file_path, input_file_name, n_splits=5):
    # Read data from the input file
    file_path = f'{input_file_path}/{input_file_name}'
    df = pd.read_excel(file_path)

    # Drop rows with missing values in the target column (POWER GENERATION)
    df = df.dropna(subset=['POWER GENERATION (MW)'])

    # Extract features and target
    X = df[['TEMPERATURE (°C)', 'WIND SPEED (m/s)', 'SOLAR RADIATION (W/m2)', 'CLOUDNESS (%)']]
    y = df['POWER GENERATION (MW)']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a neural network model
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=19)

    # Create a cross-validator (e.g., KFold)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=19)

    # Perform cross-validation
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=make_scorer(mean_squared_error))

    # Print the mean and standard deviation of the cross-validation score
    print(f'Cross-Validation Mean Squared Error: {scores.mean()}')
    print(f'Cross-Validation Std Dev of Squared Error: {scores.std()}')

    # Forecast power generation for the next 7 days (168 hours)
    forecast_date = df['DATE'].max() + timedelta(days=1)
    forecast_hours = pd.date_range(start=forecast_date, periods=168, freq='h')

    # Prepare features for the forecast (last 24 hours of available data)
    forecast_features = df[['TEMPERATURE (°C)', 'WIND SPEED (m/s)', 'SOLAR RADIATION (W/m2)', 'CLOUDNESS (%)']].tail(24)
    forecast_features_scaled = scaler.transform(forecast_features)

    # Fit model on the entire dataset for final prediction
    model.fit(X_scaled, y)

    # Initialize a list to hold the forecast results
    forecast = []

    for i in range(168):
        # Make predictions for each hour in the forecast range
        if i < len(forecast_features_scaled):
            prediction = model.predict(forecast_features_scaled[i:i + 1])
        else:
            # For hours beyond the initial data, you may need to implement a method to predict the new features
            # Here, we'll simply use the last available features for the next predictions
            prediction = model.predict(forecast_features_scaled[-1].reshape(1, -1))

        # Set negative values in the forecast to zero
        prediction[prediction < 0] = 0
        forecast.append(prediction[0])

        # Update forecast features for the next prediction
        forecast_features_scaled = np.append(forecast_features_scaled, prediction.reshape(1, -1), axis=0)[1:]

    # Create a DataFrame for the forecast results
    forecast_df = pd.DataFrame({'DATE': forecast_hours, 'POWER GENERATION (MW)': forecast})

    # Set the date and time format
    forecast_df['DATE'] = forecast_df['DATE'].dt.strftime('%m/%d/%Y %H:%M')

    # Plotting the forecast results
    plt.figure(figsize=(14, 7))
    plt.plot(forecast_df['DATE'], forecast_df['POWER GENERATION (MW)'], marker='o', linestyle='-', color='b')
    plt.title('7-Day Power Generation Forecast', fontsize=16)
    plt.xlabel('Date and Time', fontsize=14)
    plt.ylabel('Power Generation (MW)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

    return forecast_df


# Assuming the input file is in the same directory as the script
input_file_path = '.'
input_file_name = 'Forecast Dataset.xlsx'
forecast_result = e2109957_altuntas_GF(input_file_path, input_file_name)
print(forecast_result)
