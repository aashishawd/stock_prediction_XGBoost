import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        # Read data with specific date parsing
        df = pd.read_csv(file_path, index_col=False, parse_dates=['Date'])
        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise ValueError("Required columns 'Date' and 'Close' not found in data")
            
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def compute_rsi(df, period=14):
    try:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, float('inf'))  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        print(f"Error computing RSI: {str(e)}")
        raise

def main():
    try:
        # Load data
        df = load_data("CSVForDate.csv")

        # Calculate Moving Averages with min_periods
        df['EMA_9'] = df['Close'].ewm(span=9, min_periods=1).mean().shift()
        df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean().shift()
        df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean().shift()
        df['SMA_15'] = df['Close'].rolling(window=15, min_periods=1).mean().shift()
        df['SMA_30'] = df['Close'].rolling(window=30, min_periods=1).mean().shift()

        # Plot Moving Averages
        fig_ma = go.Figure()
        fig_ma.add_traces([
            go.Scatter(x=df['Date'], y=df['EMA_9'], name='EMA 9'),
            go.Scatter(x=df['Date'], y=df['SMA_5'], name='SMA 5'),
            go.Scatter(x=df['Date'], y=df['SMA_10'], name='SMA 10'),
            go.Scatter(x=df['Date'], y=df['SMA_15'], name='SMA 15'),
            go.Scatter(x=df['Date'], y=df['SMA_30'], name='SMA 30'),
            go.Scatter(x=df['Date'], y=df['Close'], name='Close', opacity=0.2)
        ])
        fig_ma.update_layout(
            title="Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        # Calculate RSI
        df['RSI'] = compute_rsi(df).fillna(50)  # Fill NaN with neutral RSI value

        # Plot RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
        fig_rsi.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),  # Set RSI range from 0 to 100
            template="plotly_dark"
        )

        # Calculate MACD
        ema_12 = df['Close'].ewm(span=12, min_periods=1).mean()
        ema_26 = df['Close'].ewm(span=26, min_periods=1).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()

        # Plot MACD
        fig_macd = make_subplots(rows=2, cols=1, subplot_titles=("Price with EMAs", "MACD"))
        fig_macd.add_traces([
            go.Scatter(x=df['Date'], y=df['Close'], name='Close', row=1, col=1),
            go.Scatter(x=df['Date'], y=ema_12, name='EMA 12', row=1, col=1),
            go.Scatter(x=df['Date'], y=ema_26, name='EMA 26', row=1, col=1),
            go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', row=2, col=1),
            go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal Line', row=2, col=1)
        ])
        fig_macd.update_layout(height=600, template="plotly_dark")

        # Show plots
        fig_ma.show()
        fig_rsi.show()
        fig_macd.show()
        print("Technical indicators plotted successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()