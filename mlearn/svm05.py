import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd


df = pd.read_csv('../data/breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# These are all of the values which we believe impact whether or not a tumor is benign or malignant
# That is why we call them "features"
X = np.array(df.drop(['class'], 1))
# These are the results (benign or malignant)
# We call this our "label." This is what we are predicting (malignant or benign)
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)
# Measure accuracy
accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)



