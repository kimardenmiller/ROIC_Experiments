import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# set fixed random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data set via Pandas
# x_import = pandas.read_csv('/Users/kimardenmiller/dropbox/tensorflow/data/x2015_noFinance.csv', header=None)
# x_data_values = x_import.values

# load data set via CSV Import Utility
import csv
data = list(csv.reader(open('../data/x2015_noFinance.csv')))
x_data_values = np.asarray(data)

# split into input (X) and output (Y) variables
x_data_feature_values = x_data_values[:, [0, 6, 8, 9, 13, 17, 18, 20]]  # Selecting the Features
x_data_features = x_data_feature_values[1:, :]  # Remove Labels
x_data_features[x_data_features == ''] = 0.0  # Remove Blanks
selected_feature_labels = x_data_feature_values[0, 1:]  # Just the Feature Labels
print('Selected Feature Label Names: \n', selected_feature_labels)
print('First few Stocks with Features, no Labels: ', '\n', x_data_features[0:2, :], ' ...')
print(np.size(x_data_features[:, 0]), 'Stocks by', np.size(x_data_features[0, :]), 'Features (with Tickers')

y_import = pandas.read_csv('../data/y20150106_noFin8.csv', header=None)
y_data_values = y_import[1:].values

x_tickers = x_data_features[:, 0]
print('x tickers: ', x_tickers)
y_tickers = y_data_values[:np.size(y_data_values[:, 0])-1, 0]
print('Total Y Tickers: ', np.size(y_tickers))
print('First few Y tickers: \n', y_tickers[0:5])
print('Last few Y tickers: \n', y_tickers[673:])

# Format Y to y = 1 (positive) and y = 0 (negative) examples
true_false_mask = np.in1d(x_tickers, y_tickers)
y_mask = np.where(true_false_mask, 1, 0)
total_positive_examples = np.size(np.nonzero(y_mask))
total_examples = np.size(y_mask)
print('Total Positive Y Ticker Example Count: ', total_positive_examples, )
print('Total Positive Y Ticker Example Count on x_tickers: ', np.size(x_tickers[np.nonzero(y_mask)]))
print('Total Y Ticker Mask Count: ', total_examples)

# Place dataset into input (X) and output (Y) variables
x_strings = x_data_features[:, 1:]  # take off tickers, as they can't be tensor'd
raw_X = x_strings.astype(np.float)  # convert strings to float
print('Training Examples: ', np.size(raw_X[:, 0]), ' x ', np.size(raw_X[0, :]), ' Features: ')

X = raw_X  # X value assigned
Y = y_mask  # Y uses the 0, 1 to show negative and positive examples
np.set_printoptions(precision=3, suppress=True)
print('First few X Training Examples with', np.size(raw_X[0, :]), 'Selected Features: \n', raw_X[0:2, :], ' ...')
num_of_features = np.size(raw_X[0, :])
print('Features for Tensor: ', num_of_features)

positive_scores = []


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(num_of_features, input_dim=num_of_features, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Dense(num_of_features * 10, init='normal', activation='relu', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Dense(num_of_features, init='normal', activation='relu', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=1.0, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', 'fbeta_score'])
    # evaluate the model
    scores = model.evaluate(X, Y)
    print('\n', "%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    # calculate predictions
    predictions = model.predict(X)
    np.set_printoptions(precision=3, suppress=True)
    print('Predictions (Fist 100): ', '\n', predictions[0:100, 0])
    print('Rows in Prediction: ', np.size(predictions[:, 0]))
    positive_predictions = np.sum(predictions[:, 0] > .5)
    print('Positive Predictions: ', positive_predictions)
    # print('Positive Prediction Pointers (Fist 20): ', '\n', [np.where(predictions[:, 0] > .5)][0:20])
    # print('Positive Prediction Tickers: (Fist 20) ', '\n', x_tickers[np.where(predictions[:, 0] > .5)][0:20])
    print('Positive Prediction Features (First 10): ', '\n',
          x_data_features[np.where(predictions[:, 0] > .5), 0:][0][0:10])
    print('Positive Prediction Ground Truth (Fist 50): ', '\n', Y[np.where(predictions[:, 0] > .5)][0:50])
    accurate_predictions = np.size(np.nonzero(Y[np.where(predictions[:, 0] > .5)]))
    positive_score = ((accurate_predictions / positive_predictions) * 100 if positive_predictions > 0 else 0)
    positive_scores.append(positive_score)
    print('Accuracy of Positive Predictions: ', '\n', "%.1f%%" % positive_score, '\n-----------------------\n')
    return model


# evaluate baseline model with standardized data set
estimators = []
estimators.append(('standardize', StandardScaler()))
class_weights = {0: 1, 1: 1}
estimators.append(['mlp', KerasClassifier(build_fn=create_baseline,
                                          nb_epoch=100,
                                          batch_size=10,
                                          # class_weight=class_weights,
                                          verbose=0)])
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
np.set_printoptions(precision=3, suppress=True)
print("Standardized (conventional): %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
print('\nAccuracy Metrics:', '\n-----------------------')
print('Accuracy Average of All Positive Predictions: %.2f%%   Standard Deviation: (%.2f%%)' % (np.asarray(positive_scores).mean(), np.asarray(positive_scores).std()))
print("Baseline Accuracy of Random Prediction: %.2f%% " % ((total_positive_examples / total_examples) * 100))

# Dec 10, 2016

# Standardized (conventional): 27.50% (32.55%)
#
# Accuracy Metrics:
# -----------------------
# Accuracy Average of All Positive Predictions: 24.78%   Standard Deviation: (5.24%)
# Baseline Accuracy of Random Prediction: 24.93%