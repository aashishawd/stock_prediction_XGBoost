import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

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

def determine_period(df):
    """Determine the appropriate period for seasonal decomposition based on data frequency"""
    try:
        # Count unique dates per year
        dates_per_year = df.groupby(df['Date'].dt.year).size()
        avg_dates_per_year = dates_per_year.mean()
        
        # If daily data (approximately 252 trading days per year)
        if 240 <= avg_dates_per_year <= 260:
            return 252
        # If weekly data
        elif 48 <= avg_dates_per_year <= 52:
            return 52
        # If monthly data
        elif 10 <= avg_dates_per_year <= 14:
            return 12
        else:
            return 252  # Default to daily data
    except Exception as e:
        print(f"Warning: Could not determine period, using default value: {str(e)}")
        return 252

def main():
    try:
        # Load data
        df = load_data("CSVForDate.csv")

        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        fig_candle.update_layout(
            title="Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        # Time series decomposition
        df_close = df[['Date', 'Close']].set_index('Date')
        period = determine_period(df)
        
        try:
            # Ensure data is regular and has no missing values
            df_close = df_close.asfreq('D').fillna(method='ffill')
            
            if len(df_close) < period * 2:
                raise ValueError(f"Insufficient data points for decomposition. Need at least {period * 2} points.")
                
            decomposition = seasonal_decompose(
                df_close['Close'],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
        except Exception as e:
            print(f"Warning: Could not perform seasonal decomposition: {str(e)}")
            print("This might be due to insufficient data points or non-regular time series.")
            return

        # Plot decomposition using Matplotlib
        plt.style.use('dark_background')
        fig_decomp, axes = plt.subplots(4, 1, figsize=(20, 8))
        decomposition.observed.plot(ax=axes[0], title="Observed")
        decomposition.trend.plot(ax=axes[1], title="Trend")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
        decomposition.resid.plot(ax=axes[3], title="Residual")
        plt.tight_layout()

        # Show plots
        fig_candle.show()
        plt.show()
        print("Candlestick and decomposition visualizations completed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()