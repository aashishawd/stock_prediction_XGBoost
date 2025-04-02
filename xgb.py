import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import types
from botocore.client import Config

import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import datetime


df = pd.read_csv("CSVForDate.csv", index_col=False)
dfp=df
#df['Date']=df['Date'].astype('datetime64')
df['Date'] = pd.to_numeric(pd.to_datetime(df['Date']))
df.head()
df['Close'] = df['Close'].shift(-1)
df = df.iloc[30:] 
df = df[:-1]      
df.index = range(len(df))

# Train, validation and test split
test_size  = 0.15
valid_size = 0.15
test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))
train_df  = df.loc[:valid_split_idx].copy()
valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.Close, name='Training'))
fig.add_trace(go.Scatter(x=valid_df.Date, y=valid_df.Close, name='Validation'))
fig.add_trace(go.Scatter(x=test_df.Date,  y=test_df.Close,  name='Test'))
fig.show()
drop_cols = ['Date', 'Open', 'Low', 'High']
train_df1 = train_df.drop(drop_cols, 1)
valid_df1 = valid_df.drop(drop_cols, 1)
test_df1  = test_df.drop(drop_cols, 1)
y_train = train_df1['Close'].copy()
# X_train = train_df.drop(['Close'], 1)
X_train = train_df['Date'].copy()
y_valid = valid_df1['Close'].copy()
# X_valid = valid_df.drop(['Close'], 1)
X_valid = valid_df['Date'].copy();
y_test  = test_df1['Close'].copy()
# X_test  = test_df.drop(['Close'], 1)
X_test  = test_df['Date'].copy();

#X_train['Date'] = X_train['Date'].apply((lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')))
X_train = np.array(X_train.tolist()).reshape((-1,1))
X_train.astype("str")
print(X_train.dtype)
# Grid search
parameters = {
    'n_estimators': [200],
    'learning_rate': [0.005],
    'max_depth': [8],
    'gamma': [0.001],
}
eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:linear', verbose=True)
clf = GridSearchCV(model, parameters, cv=3)
clf.fit(X_train, y_train)
print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

# Building the model
model = xgb.XGBRegressor(**clf.best_params_, objective='reg:linear')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
plot_importance(model);

# Prediction
y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:10]}')
print(f'y_pred = {y_pred[:10]}')
predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred
fig = make_subplots(rows=2, cols=1)
#dt.Date = df.Date.astype(np.datetime64)
fig.add_trace(go.Scatter(x=df.Date, y=df.Close,
                         name='Prediction',
                         marker_color='LightSkyBlue'), row=1, col=1)

# fig.add_trace(go.Scatter(x=predicted_prices.Date,
#                          y=predicted_prices.Close,
#                          name='Prediction',
#                          marker_color='MediumPurple'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=y_test,
                         name='Prediction',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

# fig.add_trace(go.Scatter(x=predicted_prices.Date,
#                          y=y_pred,
#                          name='Prediction',
#                          marker_color='MediumPurple',
#                          showlegend=False), row=2, col=1)

fig.show()
