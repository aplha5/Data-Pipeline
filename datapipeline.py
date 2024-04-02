import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from ta import add_all_ta_features
import sqlite3

def ingest_data(file_path, file_format):
    if file_format == 'csv':
        data = pd.read_csv(file_path)
    elif file_format == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")
    return data

def validate_data(data):
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        data = handle_missing_values(data)

    # Check for outliers
    data = handle_outliers(data)

    # Ensure consistency in timestamps or date formats
    data['Date'] = pd.to_datetime(data['Date'])

    return data

def handle_missing_values(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    return data

def handle_outliers(data):
    clf = IsolationForest(contamination=0.1)
    data['Outlier'] = clf.fit_predict(data[['Open', 'High', 'Low', 'Close']])
    data = data[data['Outlier'] == 1].copy()  # Ensure we're working with a copy
    data.drop(columns=['Outlier'], inplace=True)
    return data

def calculate_technical_indicators(data):
  try:
    if len(data) < 10:
        raise ValueError("Insufficient data points for calculating technical indicators.")
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
  except (ValueError, IndexError) as e:
    print(e)
  return data

def resample_data(data, freq):
    data.set_index('Date', inplace=True)
    data_resampled = data.resample(freq).agg({
        'Open': 'first', 
        'High': 'max', 
        'Low': 'min', 
        'Close': 'last', 
        'Volume': 'sum'
    }).dropna()
    return data_resampled

def store_data(data, db_name):
    conn = sqlite3.connect(db_name)
    data.to_sql('OHLC', conn, if_exists='replace', index=False)
    conn.close()

def main():
    # Step 1: Data Ingestion
    file_path = 'test.csv'
    file_format = 'csv'
    data = ingest_data(file_path, file_format)

    # Step 2: Data Validation
    data = validate_data(data)

    # Step 3: Data Transformation
    data = calculate_technical_indicators(data)
    freq = 'D'  # Daily resampling
    data_resampled = resample_data(data, freq)

    # Step 4: Data Storage
    db_name = 'OHLC_data.db'
    store_data(data_resampled, db_name)

if __name__ == "__main__":
    main()
