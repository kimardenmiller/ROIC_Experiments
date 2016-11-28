import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data set
# x_import = pandas.read_csv('/Users/kimardenmiller/dropbox/tensorflow/data/x2015_noFinance.csv', header=None)
# x_data_values = x_import.values

import csv
data = list(csv.reader(open('/Users/kimardenmiller/dropbox/tensorflow/data/x2015_noFinance.csv')))
x_data_values = np.asarray(data)

# split into input (X) and output (Y) variables
x_data_feature_values = x_data_values[:, [0, 6, 8, 9, 13, 17, 18, 20]]
x_data_features = x_data_feature_values[1:, :]   # Remove Labels
x_data_features[x_data_features == ''] = 0.0    # Remove Blanks
selected_feature_labels = x_data_feature_values[0, 1:]
print('Selected Feature Names: \n', selected_feature_labels)
print('First few Stocks with Features, no Labels: ', '\n', x_data_features[0:2, :], ' ...')
print(np.size(x_data_features[:, 0]), 'Stocks by', np.size(x_data_features[0, :]), 'Features (with Tickers')

y_import = pandas.read_csv('/Users/kimardenmiller/dropbox/tensorflow/data/y201501_noFinancials.csv', header=None)
y_data_values = y_import[1:].values

x_tickers = x_data_features[:, 0]
print('x tickers: ', x_tickers)
y_tickers = y_data_values[:, 0]
print('Total Y Tickers: ', np.size(y_tickers))
print('First few Y tickers: \n', y_tickers[0:5])

# Format Y to y = 1 (positive) and y = 0 (negative) examples
true_false_mask = np.in1d(x_tickers, y_tickers)
y_mask = np.where(true_false_mask, 1, 0)
print('Total Positive Y Ticker Example Count: ', np.size(np.nonzero(y_mask)), )
print('Total Positive Y Ticker Example Count on x_tickers: ', np.size(x_tickers[np.nonzero(y_mask)]))
print('Total Y Ticker Mask Count: ', np.size(y_mask))
print('Y after Encoding: ', y_mask[0:100])

# Place dataset into input (X) and output (Y) variables
x_strings = x_data_features[:, 1:]  # take off tickers, as they can't be tensor'd
raw_X = x_strings.astype(np.float)  # convert strings to float
print('Training Examples: ', np.size(raw_X[:, 0]), ' x ', np.size(raw_X[0, :]), ' Features: ')

X = raw_X       # X value assigned
Y = y_mask      # Y uses the 0, 1 to show negative and positive examples
np.set_printoptions(precision=3, suppress=True)
print('First few X Training Examples with', np.size(raw_X[0, :]), '\nSelected Features: ', raw_X[0:2, :], ' ...')


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# evaluate model with standardized data set
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=2)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Nov 26, 2016  Results: 9.17% (27.50%)
# Nov 27, 2016  Results: 91.71% (0.18%) (Using CSV import)