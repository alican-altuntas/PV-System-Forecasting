from datetime import timedelta
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def forecast(input_file_path, input_file_name, n_splits=5):
    # Read data from the input file
    file_path = 'Forecast Dataset.xlsx'
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

    # Print the mean and standard deviation of the cross-validation score real values and in percentage
    # to see how accurate the system
    mse = scores.mean()
    std_dev_squared_error = scores.std()
    print(f'Cross-Validation Mean Squared Error: {scores.mean()}')
    print(f'Cross-Validation Std Dev of Squared Error: {scores.std()}')

    # Forecast power generation for the next day (24 hours)
    forecast_date = df['DATE'].max() + timedelta(days=1)
    forecast_hours = pd.date_range(start=forecast_date, periods=24, freq='h')

    # Prepare features for the forecast
    forecast_features = df[['TEMPERATURE (°C)', 'WIND SPEED (m/s)', 'SOLAR RADIATION (W/m2)', 'CLOUDNESS (%)']].tail(24)
    forecast_features_scaled = scaler.transform(forecast_features)

    # Make predictions for the forecast
    forecast = model.fit(X_scaled, y).predict(forecast_features_scaled)

    # Set negative values in the forecast to zero
    forecast[forecast < 0] = 0

    # Create a DataFrame for the forecast results
    forecast_df = pd.DataFrame({'DATE': forecast_hours, 'POWER GENERATION (MW)': forecast})

    # Set the date and time format
    forecast_df['DATE'] = forecast_df['DATE'].dt.strftime('%m/%d/%Y %H:%M')

    # Filter data for specified times
    data_00 = df[df['TIME'] == '24:00']
    data_01 = df[df['TIME'] == '01:00']
    data_02 = df[df['TIME'] == '02:00']
    data_03 = df[df['TIME'] == '03:00']
    data_04 = df[df['TIME'] == '04:00']
    data_05 = df[df['TIME'] == '05:00']
    data_06 = df[df['TIME'] == '06:00']
    data_07 = df[df['TIME'] == '07:00']
    data_08 = df[df['TIME'] == '08:00']
    data_09 = df[df['TIME'] == '09:00']
    data_10 = df[df['TIME'] == '10:00']
    data_11 = df[df['TIME'] == '11:00']
    data_12 = df[df['TIME'] == '12:00']
    data_13 = df[df['TIME'] == '13:00']
    data_14 = df[df['TIME'] == '14:00']
    data_15 = df[df['TIME'] == '15:00']
    data_16 = df[df['TIME'] == '16:00']
    data_17 = df[df['TIME'] == '17:00']
    data_18 = df[df['TIME'] == '18:00']
    data_19 = df[df['TIME'] == '19:00']
    data_20 = df[df['TIME'] == '20:00']
    data_21 = df[df['TIME'] == '21:00']
    data_22 = df[df['TIME'] == '22:00']
    data_23 = df[df['TIME'] == '23:00']

    # Sum the power generation values for each day at specified time
    sum_00 = data_00.groupby(data_00['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_01 = data_01.groupby(data_01['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_02 = data_02.groupby(data_02['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_03 = data_03.groupby(data_03['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_04 = data_04.groupby(data_04['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_05 = data_05.groupby(data_05['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_06 = data_06.groupby(data_06['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_07 = data_07.groupby(data_07['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_08 = data_08.groupby(data_08['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_09 = data_09.groupby(data_09['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_10 = data_10.groupby(data_10['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_11 = data_11.groupby(data_11['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_12 = data_12.groupby(data_12['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_13 = data_13.groupby(data_13['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_14 = data_14.groupby(data_14['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_15 = data_15.groupby(data_15['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_16 = data_16.groupby(data_16['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_17 = data_17.groupby(data_17['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_18 = data_18.groupby(data_18['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_19 = data_19.groupby(data_19['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_20 = data_20.groupby(data_20['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_21 = data_21.groupby(data_21['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_22 = data_22.groupby(data_22['DATE'].dt.date)['POWER GENERATION (MW)'].sum()
    sum_23 = data_23.groupby(data_23['DATE'].dt.date)['POWER GENERATION (MW)'].sum()

    if sum_00.any() == 0:
        # If the sum is zero, set the forecast at 00:00 to zero
        forecast_df.loc[forecast_hours.hour == 0, 'POWER GENERATION (MW)'] = 0

    if sum_01.any() == 0:
        # If the sum is zero, set the forecast at 01:00 to zero
        forecast_df.loc[forecast_hours.hour == 1, 'POWER GENERATION (MW)'] = 0

    if sum_02.any() == 0:
        # If the sum is zero, set the forecast at 02:00 to zero
        forecast_df.loc[forecast_hours.hour == 2, 'POWER GENERATION (MW)'] = 0

    if sum_03.any() == 0:
        # If the sum is zero, set the forecast at 03:00 to zero
        forecast_df.loc[forecast_hours.hour == 3, 'POWER GENERATION (MW)'] = 0

    if sum_04.any() == 0:
        # If the sum is zero, set the forecast at 04:00 to zero
        forecast_df.loc[forecast_hours.hour == 4, 'POWER GENERATION (MW)'] = 0

    if sum_05.any() == 0:
        # If the sum is zero, set the forecast at 05:00 to zero
        forecast_df.loc[forecast_hours.hour == 5, 'POWER GENERATION (MW)'] = 0

    if sum_06.any() == 0:
        # If the sum is zero, set the forecast at 06:00 to zero
        forecast_df.loc[forecast_hours.hour == 6, 'POWER GENERATION (MW)'] = 0

    if sum_07.any() == 0:
        # If the sum is zero, set the forecast at 07:00 to zero
        forecast_df.loc[forecast_hours.hour == 7, 'POWER GENERATION (MW)'] = 0

    if sum_08.any() == 0:
        # If the sum is zero, set the forecast at 08:00 to zero
        forecast_df.loc[forecast_hours.hour == 8, 'POWER GENERATION (MW)'] = 0

    if sum_09.any() == 0:
        # If the sum is zero, set the forecast at 09:00 to zero
        forecast_df.loc[forecast_hours.hour == 9, 'POWER GENERATION (MW)'] = 0

    if sum_10.any() == 0:
        # If the sum is zero, set the forecast at 10:00 to zero
        forecast_df.loc[forecast_hours.hour == 10, 'POWER GENERATION (MW)'] = 0

    if sum_11.any() == 0:
        # If the sum is zero, set the forecast at 11:00 to zero
        forecast_df.loc[forecast_hours.hour == 11, 'POWER GENERATION (MW)'] = 0

    if sum_12.any() == 0:
        # If the sum is zero, set the forecast at 12:00 to zero
        forecast_df.loc[forecast_hours.hour == 12, 'POWER GENERATION (MW)'] = 0

    if sum_13.any() == 0:
        # If the sum is zero, set the forecast at 13:00 to zero
        forecast_df.loc[forecast_hours.hour == 13, 'POWER GENERATION (MW)'] = 0

    if sum_14.any() == 0:
        # If the sum is zero, set the forecast at 14:00 to zero
        forecast_df.loc[forecast_hours.hour == 14, 'POWER GENERATION (MW)'] = 0

    if sum_15.any() == 0:
        # If the sum is zero, set the forecast at 15:00 to zero
        forecast_df.loc[forecast_hours.hour == 15, 'POWER GENERATION (MW)'] = 0

    if sum_16.any() == 0:
        # If the sum is zero, set the forecast at 16:00 to zero
        forecast_df.loc[forecast_hours.hour == 16, 'POWER GENERATION (MW)'] = 0

    if sum_17.any() == 0:
        # If the sum is zero, set the forecast at 17:00 to zero
        forecast_df.loc[forecast_hours.hour == 17, 'POWER GENERATION (MW)'] = 0

    if sum_18.any() == 0:
        # If the sum is zero, set the forecast at 18:00 to zero
        forecast_df.loc[forecast_hours.hour == 18, 'POWER GENERATION (MW)'] = 0

    if sum_19.any() == 0:
        # If the sum is zero, set the forecast at 19:00 to zero
        forecast_df.loc[forecast_hours.hour == 19, 'POWER GENERATION (MW)'] = 0

    if sum_20.any() == 0:
        # If the sum is zero, set the forecast at 20:00 to zero
        forecast_df.loc[forecast_hours.hour == 20, 'POWER GENERATION (MW)'] = 0

    if sum_21.any() == 0:
        # If the sum is zero, set the forecast at 21:00 to zero
        forecast_df.loc[forecast_hours.hour == 21, 'POWER GENERATION (MW)'] = 0

    if sum_22.any() == 0:
        # If the sum is zero, set the forecast at 22:00 to zero
        forecast_df.loc[forecast_hours.hour == 22, 'POWER GENERATION (MW)'] = 0

    if sum_23.any() == 0:
        # If the sum is zero, set the forecast at 23:00 to zero
        forecast_df.loc[forecast_hours.hour == 23, 'POWER GENERATION (MW)'] = 0

    return forecast_df


# Assuming the input file is in the same directory as the script
input_file_path = '.'
input_file_name = 'Forecast Dataset.xlsx'
forecast_result = forecast(input_file_path, input_file_name)
print(forecast_result)
