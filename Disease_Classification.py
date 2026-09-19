# import a immportant library
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import pandas as pd

# read CSV File
my_File = pd.read_csv("disease_symptom_600_text.csv")
my_File.head()

symptom_cols = ["Symptom_1","Symptom_2","Symptom_3","Symptom_4","Symptom_5","Symptom_6","Symptom_7","Symptom_8"]

symptoms = my_File[symptom_cols].apply(
    lambda row: row.dropna().tolist(),
    axis=1
)
mlb = preprocessing.MultiLabelBinarizer()
X = mlb.fit_transform(symptoms)

Y = my_File[["Disease"]].values

# train and test split
train_x, test_x, train_y, test_y = train_test_split(X, Y)
print(X.shape)
print(Y.shape)

# make and test model
model = DecisionTreeClassifier(criterion="gini").fit(train_x, train_y)

yhat = model.predict(test_x)
print("score:", metrics.accuracy_score(test_y, yhat))
