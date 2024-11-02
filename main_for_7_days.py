import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from datetime import timedelta


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

    # Fit the model on the full training data
    model.fit(X_scaled, y)

    # Forecast power generation for the next 7 days (24 hours each day)
    forecast_date = df['DATE'].max() + timedelta(days=1)
    forecast_hours = pd.date_range(start=forecast_date, periods=7 * 24, freq='h')

    # Prepare features for the forecast
    last_24_hours_features = df[
        ['TEMPERATURE (°C)', 'WIND SPEED (m/s)', 'SOLAR RADIATION (W/m2)', 'CLOUDNESS (%)']].tail(24)
    last_24_hours_scaled = scaler.transform(last_24_hours_features)

    # Prepare a DataFrame to hold the forecast values
    forecast_values = []

    # Iterate over the next 7 days (168 hours)
    for hour in range(7 * 24):
        if hour < len(last_24_hours_scaled):
            prediction = model.predict(last_24_hours_scaled[hour].reshape(1, -1))
        else:
            # Use the last available features for the next predictions
            prediction = model.predict(last_24_hours_scaled[-1].reshape(1, -1))

        # Append the prediction, ensuring no negative values
        forecast_values.append(max(prediction[0], 0))

        # Prepare the next input feature set for the model
        # Instead of reshaping prediction to (1, -1), we need to create an array with the same shape as last_24_hours_scaled
        new_features = np.array(
            [[max(prediction[0], 0), 0, 0, 0]])  # Adjust according to the features (keep others as placeholders)

        if hour < len(last_24_hours_scaled) - 1:
            # Append the new features for the next prediction
            last_24_hours_scaled = np.append(last_24_hours_scaled, new_features, axis=0)[1:]

    # Create a DataFrame for the forecast results
    forecast_df = pd.DataFrame({'DATE': forecast_hours, 'POWER GENERATION (MW)': forecast_values})

    # Set the date and time format
    forecast_df['DATE'] = forecast_df['DATE'].dt.strftime('%m/%d/%Y %H:%M')

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df['DATE'], forecast_df['POWER GENERATION (MW)'], marker='o', linestyle='-', color='b')
    plt.title('7-Day Power Generation Forecast')
    plt.xlabel('Date and Time')
    plt.ylabel('Power Generation (MW)')
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
