# Photovoltaic (PV) Generation Forecasting on an Hourly Basis
 Project Description:
     This project aims to forecast the hourly photovoltaic (PV) generation for the upcoming day using a given hourly basis data sets. The forecasting model leverages weather data, historical PV generation data, and other relevant parameters to predict future energy production. This can be useful for optimizing energy usage, grid management, and planning maintenance.

Features:
    - Predicts PV generation for the next 7 days on an hourly basis.
    - Uses weather data such as temperature, solar radiation, and cloud cover for better accuracy.
    - Based on machine learning techniques (mention specific algorithm if applicable).
    - Easy integration with energy management systems.
 Installation & Setup:
    1. Clone the Repository: Open your terminal and run:
        git clone https://github.com/alican-altuntas/PV-System-Forecasting.git
        cd PV-System-Forecasting
    2. Install Dependencies: Make sure Python is installed on your system. Then, install the required Python packages using:
        pip install -r requirements.txt
    3. Set Up the Dataset: Download the dataset from the provided link or use your own data. The data should contain the following columns:
        Date: DD/MM/YYYY
        Time: HH:MM
        Temperature: XX.X (°C)
        Wind Speed: XX.X (m/s)
        Solar Radiation: XX.X (W/m²)
        Cloudness: XX.X (%)
        Power Generation: XX.X (MW)
      Place the .xlsx dataset file in the same folder with the main.py folder
      

