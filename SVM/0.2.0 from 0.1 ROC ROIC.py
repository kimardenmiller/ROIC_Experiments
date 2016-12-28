
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X, y = X[y != 2], y[y != 2]
# n_samples, n_features = X.shape

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

import pandas
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
y = y_mask  # Y uses the 0, 1 to show negative and positive examples
np.set_printoptions(precision=3, suppress=True)
print('First few X Training Examples with', np.size(raw_X[0, :]), 'Selected Features: \n', raw_X[0:2, :], ' ...')
num_of_features = np.size(raw_X[0, :])
print('Features for Tensor: ', num_of_features)

print("Starting ...")

# Add noisy features
random_state = np.random.RandomState(0)
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear',
                     probability=True,
                     random_state=random_state)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

positive_scores = []

i = 0
for (train, test), color in zip(cv.split(X, y), colors):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])

    # calculate predictions
    predictions = classifier.predict(X)
    np.set_printoptions(precision=3, suppress=True)
    print('Predictions (Fist 100): ', '\n', predictions[0:100])
    print('Rows in Prediction: ', np.size(predictions))
    positive_predictions = np.sum(predictions > .5)
    print('Positive Predictions: ', positive_predictions)
    # print('Positive Prediction Pointers (Fist 20): ', '\n', [np.where(predictions[:, 0] > .5)][0:20])
    # print('Positive Prediction Tickers: (Fist 20) ', '\n', x_tickers[np.where(predictions[:, 0] > .5)][0:20])
    print('Positive Prediction Features (First 10): ', '\n',
          x_data_features[np.where(predictions > .5), 0:][0][0:10])
    print('Positive Prediction Ground Truth (Fist 50): ', '\n', y[np.where(predictions > .5)][0:50])
    accurate_predictions = np.size(np.nonzero(y[np.where(predictions > .5)]))
    positive_score = ((accurate_predictions / positive_predictions) * 100 if positive_predictions > 0 else 0)
    positive_scores.append(positive_score)
    print('Accuracy of Positive Predictions: ', '\n', "%.1f%%" % positive_score, '\n-----------------------\n')

    # Compute ROC curve and area the curve
    # fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    # mean_tpr += interp(mean_fpr, fpr, tpr)
    # mean_tpr[0] = 0.0
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=lw, color=color,
    #          label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1

# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')
#
# mean_tpr /= cv.get_n_splits(X, y)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic on ROIC')
# plt.legend(loc="lower right")
# plt.show()

print('\nAccuracy Metrics:', '\n-----------------------')
print('Accuracy Average of All Positive Predictions: %.2f%%   Standard Deviation: (%.2f%%)' % (np.asarray(positive_scores).mean(), np.asarray(positive_scores).std()))
print("Baseline Accuracy of Random Prediction: %.2f%% " % ((total_positive_examples / total_examples) * 100))

