import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

heart = pd.read_csv("E:\\Datasets\\DataSet\\train_2v.csv")

heart['hypertension'] = heart['hypertension'].astype('float')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lencoder = LabelEncoder()
heart.iloc[:, 1:2] = lencoder.fit_transform(heart.iloc[:, 1:2])
heart.iloc[:, 5:6] = lencoder.fit_transform(heart.iloc[:, 5:6])

from sklearn.preprocessing import Imputer
imputer_mean = Imputer(missing_values='NaN', strategy="mean", axis=0)
imputer_most_frequent = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
heart.iloc[:, 9:10]= imputer_mean.fit_transform(heart.iloc[:, 9:10])


x= heart.iloc[:, [2,4,5]]
y= heart.iloc[:, 11]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)
y_pred = classifier.predict(x_test)

pickle.dump(classifier, open('model.pkl','wb'))
x

