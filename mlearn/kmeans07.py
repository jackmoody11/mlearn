"""
PClass Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
survival Survival (0 = No, 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses aboard
parch Number of Parents/Children aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd

style.use('ggplot')


def handle_non_numeric_data(d):
    columns = d.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]
        if d[column].dtype != np.int64 and d[column].dtype != np.float64:
            column_contents = d[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            d[column] = list(map(convert_to_int, d[column]))
    return d


df = pd.read_excel('../data/titanic.xls')
df.drop(['body', 'name', 'ticket', 'boat'], 1, inplace=True)
# df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(0, inplace=True)
df = handle_non_numeric_data(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))

