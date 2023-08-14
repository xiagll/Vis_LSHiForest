#! /usr/bin/python

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from iffad.detectors import VSSampling


from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH

import warnings

warnings.filterwarnings("ignore")

plot_num = 1

rng = np.random.RandomState(42)

datasets = ['data_twospirals']

for data_str in datasets:
    data = pd.read_csv('data/' + data_str + '.csv', header=None)
    X = data.values[:, :-1]
    ground_truth = data.values[:, -1].tolist()
    n_samples = len(ground_truth)
    outliers_fraction = 0.02

    for i in range(len(ground_truth)):
        if ground_truth[i] == 1.0:
            ground_truth[i] = False
        else:
            ground_truth[i] = True
    # print ground_truth
    mx = np.amax(X, axis=0)
    mm = np.amin(X, axis=0)

    num_ensemblers = 100
    threshold = 500

    classifiers = [("iForest", IsolationForest(random_state=rng, n_estimators=num_ensemblers)),
                   ("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH(), 1)),
                   ("L1SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1), 1)),
                   ("L2SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=2), 1)),
                   ("KLSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(3, 'rbf', 0.1), 1))]

    # Compare given classifiers under given settings
    x_left = mm[0] - 0.05 * abs(mm[0])
    x_right = mx[0] + 0.05 * abs(mx[0])
    y_bottom = mm[1] - 0.05 * abs(mm[1])
    y_top = mx[1] + 0.02 * abs(mx[1])

    xx, yy = np.meshgrid(np.linspace(x_left, x_right, 50), np.linspace(y_bottom, y_top, 50))
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)

    np.random.seed(42)

    # Fit the model
    plt.figure(figsize=(15, 15 / (len(classifiers) + 1)))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01)

    ## Plot the original twospirals
    subplot = plt.subplot(len(datasets), len(classifiers) + 1, plot_num)
    subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], marker='o', c='white', edgecolors='black')
    subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], marker='o', c='black')
    subplot.axis('tight')
    subplot.set_title('twospirals')
    subplot.set_xlim((x_left, x_right))
    subplot.set_ylim((y_bottom, y_top))
    subplot.axis('off')

    for i, (clf_name, clf) in enumerate(classifiers):
        start_time = time.time()
        print(clf_name)
        clf.fit(X)

        y_pred = clf.decision_function(X).ravel()
        auc = roc_auc_score(ground_truth, y_pred)
        print('AUC:', auc)

        # print y_pred
        threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)
        # print threshold

        y_pred = y_pred > threshold
        # false_positive=0
        # false_negative=0
        # for j in range(len(ground_truth)):
        #	if y_pred[j] and not ground_truth[j]:
        #		false_positive += 1
        #	if not y_pred[j] and ground_truth[j]:
        #		false_negative += 1
        n_errors = (y_pred != ground_truth).sum()
        print('Detected anomalies: ', n_errors)
        print('time: ', time.time() - start_time)

        # produce the anomalies detected by LSHiForest
        anomalies = []
        for i in range(len(y_pred)):
            if y_pred[i] == False:
                anomalies.append(X[i])
        anomalies = np.array(anomalies)

        # br = clf.get_avg_branch_factor()
        # if isinstance(clf, IsolationForest):
        #     print('branching_factor: ', br)

        # plot the levels lines and the points
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plot_num += 1
        subplot = plt.subplot(len(datasets), len(classifiers) + 1, plot_num)
        # plt.title(clf_name, size=18)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), int(max(
            [abs(x_left), abs(x_right), abs(y_bottom), abs(y_top)]))), cmap=plt.cm.Blues_r)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, int(max(
            [abs(x_left), abs(x_right), abs(y_bottom), abs(y_top)]))), cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, linestyle='-', colors='black')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
        b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')
        c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black')
        d = subplot.scatter(anomalies[:, 0], anomalies[:, 1], c='red')
        subplot.axis('tight')
        subplot.set_title("%s: (%.2f, %d)" % (clf_name, auc, n_errors))

        # subplot.legend([a.collections[0], b, c], ['learned decision function', 'true inliers', 'true outliers'], prop=matplotlib.font_manager.FontProperties(size=11), loc='best')
        # aa = "r ≈"
        # subplot.set_title("%s: %.2f" % (clf_name, auc))
        # subplot.set_title("%s (%s %.2f): %.2f" % (clf_name, aa, br, auc))
        # subplot.set_title("%s (%.2f): (%.2f, %d)" % (clf_name, br, auc, n_errors))
        # subplot.set_title("(%c) %s (%.2f): (%.2f, %d)" % (chr(ord('b')+i), clf_name, br, auc, n_errors))
        subplot.set_xlim((x_left, x_right))
        subplot.set_ylim((y_bottom, y_top))
        subplot.axis('off')
        # plt.subplots_adjust(0.04, 0.1, 0.96, 0.92, 0.1, 0.26)

plt.savefig('./twospirals_l2sh.pdf', format='pdf')
plt.show()
