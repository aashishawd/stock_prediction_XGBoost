from flask import Flask, request, render_template, flash, redirect, url_for
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.utils
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

@app.route('/')
def home():
    return render_template('design.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot_prediction():
    try:
        # Load data based on user input
        company_file = request.form['companyname']
        if not os.path.exists(company_file):
            flash('Error: Selected company file not found.', 'error')
            return redirect(url_for('predict'))
            
        # Read data with specific date parsing
        df = pd.read_csv(company_file, index_col=False, parse_dates=['Date'])
        if 'Date' not in df.columns or 'Close' not in df.columns:
            flash('Error: Invalid data format. Required columns missing.', 'error')
            return redirect(url_for('predict'))
            
        # Convert date to numeric (timestamp) and handle any errors
        df['Date_numeric'] = pd.to_numeric(df['Date'], errors='coerce')
        if df['Date_numeric'].isna().any():
            flash('Error: Invalid date format in the data.', 'error')
            return redirect(url_for('predict'))
            
        # Shift close prices and handle NaN values
        df['Close'] = df['Close'].shift(-1)
        df = df.dropna(subset=['Close', 'Date_numeric'])
        df = df.iloc[30:-1].reset_index(drop=True)

        # Split data
        test_size, valid_size = 0.15, 0.15
        test_idx = int(df.shape[0] * (1 - test_size))
        valid_idx = int(df.shape[0] * (1 - (valid_size + test_size)))
        train_df = df[:valid_idx + 1]
        valid_df = df[valid_idx + 1:test_idx + 1]
        test_df = df[test_idx + 1:]

        # Prepare features and target
        X_train = train_df['Date_numeric'].values.reshape(-1, 1)
        y_train = train_df['Close'].values
        X_valid = valid_df['Date_numeric'].values.reshape(-1, 1)
        y_valid = valid_df['Close'].values
        X_test = test_df['Date_numeric'].values.reshape(-1, 1)
        y_test = test_df['Close'].values

        # XGBoost with GridSearchCV
        params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.005, 0.001],
            'max_depth': [6, 8, 10],
            'gamma': [0.001, 0.01, 0.1]
        }
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        model = xgb.XGBRegressor(objective='reg:squarederror', verbose=False)
        clf = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)
        best_model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
        best_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Predict
        y_pred = best_model.predict(X_test)

        # Plotting
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Historical Prices", "Predictions vs Actual"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historical', marker_color='LightSkyBlue'), row=1, col=1)
        fig.add_trace(go.Scatter(x=test_df['Date'], y=y_test, name='Actual', marker_color='LightSkyBlue'), row=2, col=1)
        fig.add_trace(go.Scatter(x=test_df['Date'], y=y_pred, name='Predicted', marker_color='MediumPurple'), row=2, col=1)
        fig.update_layout(height=600, showlegend=True)

        # Render plot in HTML
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('display.html', graphJSON=graph_json)
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('predict'))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)