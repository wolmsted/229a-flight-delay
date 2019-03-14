from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import pandas as pd
import numpy as np


train = pd.read_csv('../model_data/train.csv', nrows=10000, header=None).sample(n=10000).values
validation = pd.read_csv('../model_data/validation.csv', nrows=10000, header=None).sample(n=10000).values
# train = pd.read_csv('../model_data/train.csv', header=None, nrows=10000).values
# validation = pd.read_csv('../model_data/validation.csv', header=None, nrows=1000).values

print 'Logistic Regression:'
print ''
model = LogisticRegression(C=0.8).fit(train[:,:-1], train[:,-1])
print 'Train accuracy:'
print model.score(train[:,:-1], train[:,-1])
print 'Validation accuracy:'
print model.score(validation[:,:-1], validation[:,-1])

y_hat = model.predict(validation[:,:-1])
print classification_report(validation[:,-1], y_hat, target_names=['On time', 'Delayed'])
print ''

print 'SVM'
model = SVC(C=0.5, kernel='linear').fit(train[:,:-1], train[:,-1])
print 'Train accuracy:'
print model.score(train[:,:-1], train[:,-1])
print 'Validation accuracy:'
print model.score(validation[:,:-1], validation[:,-1])

y_hat = model.predict(validation[:,:-1])
print classification_report(validation[:,-1], y_hat, target_names=['On time', 'Delayed'])
