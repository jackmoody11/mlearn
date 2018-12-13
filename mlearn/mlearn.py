import datetime
import math
import os
import pickle
import quandl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


# Choose styling for plotting
style.use('ggplot')

quandl.ApiConfig.api_key = os.environ['quandl_api_key']
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]

# Set high - low percentage and daily percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) * 100 / df['Adj. Close']
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) * 100 / df['Adj. Open']

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
# Fill missing data with outliers (to be ignored)
df.fillna(-99999, inplace=True)
# Set number of days to forecast out
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
# Set X to array of DataFrame without label column
X = np.array(df.drop(['label'], 1))
# Center to mean and scale to unit variance
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

# Drop rows with missing data
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
with open('data/linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('data/linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
