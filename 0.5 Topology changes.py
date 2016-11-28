
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[2]:

print("Initial Feature Names: \n ['total yield' 'ms debt' 'roic v2' 'book price' 'sic' 'opInc' 'fcf yield'\n 'comStockEq' 'ms cem' 'total return' 'spitz roic' 'ms ce' 'momentum'] \n [0,6,7,8,9,10,12,13,15,16,17,18,19,20]")


# In[3]:

# Import all tradable stocks of 2015 (features X)
import numpy as np
import csv
data = list(csv.reader(open('/Users/kimardenmiller/dropbox/tensorflow/data/x2015_noFinance.csv')))
feature_data = np.asarray(data)
# print('Example Stock with All Features: ', '\n', feature_data[0:2,:], ' ...')  # First stock in X
# First stock in X with selected features
selected_features = feature_data[:, [0, 6, 8, 9, 13, 17, 18, 20]]
# print('Example Stock with Selected Features: ', '\n', selected_features[0:2,:], ' ...')
# X with selected features and labels & blanks removed
x_data_features = selected_features[1:,:]
x_data_features[x_data_features == ''] = 0.0
x_data = x_data_features
selected_feature_labels = selected_features[0,1:]
print('Selected Feature Names: \n', selected_feature_labels)
print('First few Stocks with Features, no Lables: ', '\n', x_data[0:2, :], ' ...')
print(np.size(x_data[:,0]), 'Stocks by', np.size(x_data[0, :]), 'Features')


# In[4]:

# Import best performing stocks of 2015 (y = 1)
import csv
data = list(csv.reader(open('/Users/kimardenmiller/dropbox/tensorflow/data/y201501_noFinancials.csv')))
y_data = np.asarray(data[1:])
print('First few Positive Examples: ', '\n', y_data[0:3,0:5], ' ...')
# Find X and Y tickers
x_tickers = x_data[:, 0]
y_tickers = y_data[:, 0]
print('Total Y Tickers: ', np.size(y_tickers))
# Format Y to y = 1 (positive) and y = 0 (negative) examples 
true_false_mask = np.in1d(x_tickers, y_tickers)
y_mask = np.where(true_false_mask, 1, 0)
print('Total Positive Y Ticker Example Count: ', np.size(np.nonzero(y_mask)), )
print('Total Positive Y Ticker Example Count on x_tickers: ', np.size(x_tickers[np.nonzero(y_mask)]))
# Place dataset into input (X) and output (Y) variables
x_strings = x_data[:, 1:]  # take off tickers, as they can't be tensor'd
raw_X = x_strings.astype(np.float)  # convert strings to float
Y = y_mask        # Y uses the 0, 1 to show negative and positive examples
np.set_printoptions(precision=3, suppress=True)
print('First few X Training Examples with', np.size(raw_X[0, :]) , 'Selected Features: \n', raw_X[0:2, :], ' ...')


# In[5]:

# A)  Standardize the X data (*** But this approach gives negatives, which is not good.)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(raw_X)
std_rescaledX = scaler.transform(raw_X)
# summarize transformed data
np.set_printoptions(precision=3, suppress=True)
print('Pre Standardizing: \n', raw_X[0:5, :])
print('After Standardizing: \n', std_rescaledX[0:5, :])


# In[5]:

# B)  Rescale data (between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
range_rescaledX = scaler.fit_transform(raw_X)
# summarize transformed data
np.set_printoptions(precision=3)
print('Pre scaling: \n', raw_X[0:5, :])
print('After scaling: \n', range_rescaledX[0:5, :])


# In[5]:

# C)  Normalize data (length of 1)
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(raw_X)
normalizedX = scaler.transform(raw_X)
# summarize transformed data
np.set_printoptions(precision=3)
print('Pre scaling: \n', raw_X[0:5, :])
print('After scaling: \n', normalizedX[0:5, :])


# In[5]:

# create model
model = Sequential()
model.add(Dense(7, input_dim=7, init='uniform', activation='relu'))
model.add(Dense(7, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# In[6]:

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'fbeta_score'])

# Set any Feature Normalization
X = raw_X # Choices: std_rescaledX, range_rescaledX, normalizedX or just raw_X

print('Pre Feature Normalization: \n', raw_X[0:5, :])
print('After Feature Normalization: \n', X[0:5, :])


# In[ ]:

# Fit the model
print('\n Please wait for Model fit . . . \n')
model.fit(X, Y, nb_epoch=900, batch_size=10, verbose=0)
# model.fit(normalizedX, Y, nb_epoch=300, batch_size=10) # Change back to 150 after testing


# In[11]:

# evaluate the model
scores = model.evaluate(X, Y)
# scores = model.evaluate(normalizedX, Y)
print('\n', "%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
# print(model.metrics_names)
# print(scores)


# In[12]:

# calculate predictions
predictions = model.predict(X)
# predictions = model.predict(normalizedX)
np.set_printoptions(precision=3, suppress=True)
print('Fist 100 Predictions: ', '\n', predictions[0:100, 0])
print('Rows in Prediction: ', np.size(predictions[:, 0]))
positive_predictions = np.sum(predictions[:, 0] > .5)
print('Positive Predictions: ', positive_predictions)
print('Positive Prediction Pointers: ', '\n', np.where(predictions[:, 0] > .5))
print('Positive Prediction Tickers: ', '\n', x_tickers[np.where(predictions[:, 0] > .5)])

picks_data = feature_data[1:,:]
print('Positive Prediction Tickers: ', '\n', picks_data[np.where(predictions[:, 0] > .5), 0:2][0])
print('Positive Prediction Ground Truth: ', '\n', Y[np.where(predictions[:, 0] > .5)])
accurate_predictions = np.size(np.nonzero(Y[np.where(predictions[:, 0] > .5)]))
print('Accuracy of Positive Predictions: ', '\n', "%.1f%%" % ((accurate_predictions / positive_predictions) * 100 if positive_predictions > 0 else 0))




