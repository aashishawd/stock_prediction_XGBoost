import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

def load_and_preprocess_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        # Read data with specific date parsing
        df = pd.read_csv(file_path, index_col=False, parse_dates=['Date'])
        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise ValueError("Required columns 'Date' and 'Close' not found in data")
            
        # Sort data by date and handle missing values
        df = df.sort_values('Date')
        df = df.dropna(subset=['Close'])
        
        if len(df) < 100:  # Minimum data points required
            raise ValueError("Insufficient data points for training")
            
        close_prices = df['Close'].values.reshape(-1, 1)
        return df, close_prices
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def prepare_sequences(data, lookback):
    x, y = [], []
    for i in range(lookback, len(data)):
        x.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def plot_results(train, valid, test, predictions, title='LSTM Stock Price Prediction'):
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(train['Date'], train['Close'], label='Training Data', color='#1f77b4')
    plt.plot(valid['Date'], valid['Close'], label='Validation Data', color='#ff7f0e')
    plt.plot(test['Date'], test['Close'], label='Actual Prices', color='#2ca02c')
    plt.plot(test['Date'], predictions, label='Predictions', color='#d62728', linestyle='--')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    try:
        # Load and preprocess data
        df, close_prices = load_and_preprocess_data("CSVForDate.csv")
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Split into training, validation and test sets
        train_len = int(np.ceil(len(close_prices) * 0.7))
        valid_len = int(np.ceil(len(close_prices) * 0.15))
        
        train_data = scaled_data[:train_len, :]
        valid_data = scaled_data[train_len:train_len+valid_len, :]
        test_data = scaled_data[train_len+valid_len:, :]

        # Prepare sequences
        lookback = 60
        x_train, y_train = prepare_sequences(train_data, lookback)
        x_valid, y_valid = prepare_sequences(valid_data, lookback)
        x_test, y_test = prepare_sequences(test_data, lookback)

        # Reshape for LSTM
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # Build and train model
        model = build_lstm_model((x_train.shape[1], 1))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )

        # Train model
        history = model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=50,
            validation_data=(x_valid, y_valid),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        # Make predictions
        predictions = model.predict(x_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
        r2 = r2_score(y_test_original, predictions)
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Visualize results
        train = df[:train_len]
        valid = df[train_len:train_len+valid_len]
        test = df[train_len+valid_len:].copy()
        test['Predictions'] = predictions

        # Plot results
        plot_results(train, valid, test, predictions)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()