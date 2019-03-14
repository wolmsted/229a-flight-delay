import pandas as pd
import numpy as np



df = pd.read_csv('data/regular.csv')
airlines = pd.get_dummies(df['Reporting_Airline'])
print airlines.shape
origins = pd.get_dummies(df['Origin'])
print origins.shape
dests = pd.get_dummies(df['Dest'])
print dests.shape
df = df.drop(['Reporting_Airline', 'Origin', 'Dest'], axis=1)
df = pd.concat([airlines, origins, dests, df], axis=1)

matrix = df.astype(float).values
rows = matrix.shape[0]
train_length = int(0.8 * rows)
validation_length = train_length + int(0.1 * rows)
train = matrix[:train_length, :]
validation = matrix[train_length:validation_length, :]
test = matrix[validation_length:, :]
np.savetxt('model_data/train_reg.csv', train, delimiter=',')
np.savetxt('model_data/validation_reg.csv', validation, delimiter=',')
np.savetxt('model_data/test_reg.csv', test, delimiter=',')

