#!/usr/bin python
#-*- coding:utf-8 -*-
'''
author:zhiqiangxu
date:2016/8/7
'''
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import time, os, random, sys
import math
import hyperopt.tpe
import hpsklearn.components
import hpsklearn.demo_support
random.seed(1)

#choosing some samples and random split into train set and test set
def datasetSplit(libSvmFile, trainFileName, testFileName, testSetRatio, lines):
    dataFile = open(libSvmFile, 'r')
    dataList = dataFile.readlines()
    totalLines = len(dataList)
    testFileLength = int(testSetRatio*lines)
    trainFileLength = lines - testFileLength
    List = range(totalLines)
    random.shuffle(List)
    trainFile = open(trainFileName, 'w')
    testFile = open(testFileName, 'w')
    posSampleCnt = 0
    for i in range(lines):
        if float(dataList[List[i]].split(' ')[0]) > 0.0:
            posSampleCnt = posSampleCnt + 1
        if i < trainFileLength:
            trainFile.write(dataList[List[i]])
        else:
            testFile.write(dataList[List[i]])
    dataFile.close()
    trainFile.close()
    testFile.close()
    print('Positive Sample Count: %d' % posSampleCnt)
    return posSampleCnt

#calculate the positive and negative samples counts
def calcPosNegCnt(libSvmFile):
    dataFile = open(libSvmFile, 'r')
    dataList = dataFile.readlines()
    posSampleCnt = 0
    negSampleCnt = 0
    for i in range(len(dataList)):
        if float(dataList[i].split(' ')[0]) > 0.0:
            posSampleCnt = posSampleCnt + 1
        else:
            negSampleCnt = negSampleCnt + 1
    print 'Positive Sample: %d' % posSampleCnt
    print 'Negative Sample: %d' % negSampleCnt

#training xgboost and using xgboost to encode test set features
def xgboost_lr_train_test(libsvmFileNameInitial):
    posSampleCnt = datasetSplit(libsvmFileNameInitial, 'data_train_th100', 'data_test_th100', 0.2, 1100000)
    X_train, y_train = load_svmlight_file('data_train_th100')
    print(X_train.shape)
    X_test, y_test = load_svmlight_file('data_test_th100')
    #training xgboost
    negPosRatio = (1100000-posSampleCnt)/posSampleCnt
    xgbclf = xgb.XGBClassifier(nthread=4, scale_pos_weight=negPosRatio, learning_rate=0.08,
                            n_estimators=120, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
    xgbclf.fit(X_train, y_train)
    y_pred_train = xgbclf.predict_proba(X_train)[:, 1]
    xgb_train_auc = roc_auc_score(y_train, y_pred_train)
    print('xgboost train auc: %.5f' % xgb_train_auc)
    y_pred_test = xgbclf.predict_proba(X_test)[:, 1]
    xgb_test_auc = roc_auc_score(y_test, y_pred_test)
    print('xgboost test auc: %.5f' % xgb_test_auc)
    #using xgboost to encode train set and test set features
    X_train_leaves = xgbclf.apply(X_train)
    train_rows = X_train_leaves.shape[0]
    X_test_leaves = xgbclf.apply(X_test)
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)
    (rows, cols) = X_leaves.shape
    cum_count = np.zeros((1, cols), dtype=np.int32)
    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(X_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(X_leaves[:, j])) + cum_count[0][j-1]
    print('Transform features genenrated by xgboost...')
    for j in range(cols):
        keyMapDict = {}
        if j == 0:
            initial_index = 1
        else:
            initial_index = cum_count[0][j-1]+1
        for i in range(rows):
            if keyMapDict.has_key(X_leaves[i, j]) == False:
                keyMapDict[X_leaves[i, j]] = initial_index
                X_leaves[i, j] = initial_index
                initial_index = initial_index + 1
            else:
                X_leaves[i, j] = keyMapDict[X_leaves[i, j]]
    #writing encoded features into file
    print('Write xgboost learned features to file ...')
    xgbFeatureLibsvm = open('xgb_feature_libsvm', 'w')
    for i in range(rows):
        if i < train_rows:
            xgbFeatureLibsvm.write(str(y_train[i]))
        else:
            xgbFeatureLibsvm.write(str(y_test[i-train_rows]))
        for j in range(cols):
            xgbFeatureLibsvm.write(' '+str(X_leaves[i, j])+':1.0')
        xgbFeatureLibsvm.write('\n')
    xgbFeatureLibsvm.close()

#using xgboost encoded feature in lr to calculate auc
def xgb_feature_lr_train_test(xgbfeaturefile, origin_libsvm_file):
    datasetSplit(origin_libsvm_file, 'data_train_th100', 'data_test_th100', 0.2, 1100000)
    datasetSplit(xgbfeaturefile, 'xgb_feature_train_libsvm','xgb_feature_test_libsvm', 0.2, 1100000)
    X_train_origin, y_train_origin = load_svmlight_file('data_train_th100')
    X_test_origin, y_test_origin = load_svmlight_file('data_test_th100')
    X_train, y_train = load_svmlight_file('xgb_feature_train_libsvm')
    print(X_train.shape)
    X_test, y_test = load_svmlight_file('xgb_feature_test_libsvm')
    print(X_test.shape)

    #fittting lr using just xgboost encoded feature
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'lr.m')
    y_pred_train = lr.predict_proba(X_train)[:, 1]
    lr_train_auc = roc_auc_score(y_train, y_pred_train)
    print('LR Train AUC: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_pred_test)
    print('LR Test AUC: %.5f' % lr_test_auc)

    # fitting lr using xgboost encoded feature and original feature
    X_train_ext = hstack([X_train_origin, X_train])
    print(X_train_ext.shape)
    del(X_train)
    del(X_train_origin)
    X_test_ext = hstack([X_test_origin, X_test])
    print(X_test_ext.shape)
    del(X_test)
    del(X_test_origin)
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr.fit(X_train_ext, y_train)
    joblib.dump(lr, 'lr_ext.m')
    y_pred_train = lr.predict_proba(X_train_ext)[:, 1]
    lr_train_auc = roc_auc_score(y_train, y_pred_train)
    print('LR Ext Train AUC: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test_ext)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_pred_test)
    print('LR Ext Test AUC: %.5f' % lr_test_auc)

#using gbdt, gbdt+lr to calculate auc
def gbdt_lr_train_test(libsvmFileName):
    datasetSplit(libsvmFileName, 0.2, 'label_feature_data_train', 'label_feature_data_test', 600000)
    X_train, y_train = load_svmlight_file('label_feature_data_train')
    X_test, y_test = load_svmlight_file('label_feature_data_test')
    gbclf = GradientBoostingClassifier(n_estimators=30, max_depth=4, verbose=0)
    tuned_parameter = [{'n_estimators':[30, 40, 50, 60], 'max_depth':[3, 4, 5, 6], 'max_features':[0.5,0.7,0.9]}]
    gs_clf = GridSearchCV(gbclf, tuned_parameter, cv=5, scoring='roc_auc')
    gs_clf.fit(X_train.toarray(), y_train)
    print('best parameters set found: ')
    print(gs_clf.best_params_)
    y_pred_gbdt = gs_clf.predict_proba(X_test.toarray())[:, 1]
    gbdt_auc = roc_auc_score(y_test, y_pred_gbdt)
    print('gbdt auc: %.5f' % gbdt_auc)
    X_train_leaves = gbclf.apply(X_train)[:,:,0]
    (train_rows, cols) = X_train_leaves.shape
    X_test_leaves = gbclf.apply(X_test)[:,:,0]
    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))
    lr = LogisticRegression()
    lr.fit(X_trans[:train_rows, :], y_train)
    y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    gbdtlr_auc1 = roc_auc_score(y_test, y_pred_gbdtlr1)
    print('gbdt+lr auc 1: %.5f' % gbdtlr_auc1)
    lr = LogisticRegression(n_jobs=-1)
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    lr.fit(X_train_ext, y_train)
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])
    y_pred_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
    gbdtlr_auc2 = roc_auc_score(y_test, y_pred_gbdtlr2)
    print('gbdt+lr auc 2: %.5f' % gbdtlr_auc2)

#using lr to calculate auc on original data and cross featured data
def lr_train_test(libsvmFileInitial, libsvmFileCross):
    datasetSplit(libsvmFileInitial, 'data_train_th500', 'data_test_th500', 0.2, 1100000)
    datasetSplit(libsvmFileCross, 'data_cross_train_th500', 'data_cross_test_th500', 0.2, 1100000)
    X_train_origin, y_train_origin = load_svmlight_file('data_train_th500')
    print(X_train_origin.shape)
    X_test_origin, y_test_origin = load_svmlight_file('data_test_th500')
    print(X_test_origin.shape)
    lr = LogisticRegression(C=0.1, penalty='l2')
    lr.fit(X_train_origin, y_train_origin)
    y_pred_train = lr.predict_proba(X_train_origin)[:, 1]
    lr_train_auc = roc_auc_score(y_train_origin, y_pred_train)
    print('lr train auc origin: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test_origin)[:, 1]
    lr_test_auc = roc_auc_score(y_test_origin, y_pred_test)
    print('lr test auc origin: %.5f' % lr_test_auc)
    X_train_cross, y_train_cross = load_svmlight_file('data_cross_train_th500')
    print(X_train_cross.shape)
    X_test_cross, y_test_cross = load_svmlight_file('data_cross_test_th500')
    print(X_test_cross.shape)
    lr = LogisticRegression(C=0.1, penalty='l2')
    lr.fit(X_train_cross, y_train_cross)
    y_pred_train = lr.predict_proba(X_train_cross)[:, 1]
    lr_train_auc = roc_auc_score(y_train_cross, y_pred_train)
    print('lr train auc cross: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test_cross)[:, 1]
    lr_test_auc = roc_auc_score(y_test_cross, y_pred_test)
    print('lr test auc cross: %.5f' % lr_test_auc)

#using hyperopt-sklearn to automatically tune the parameters of gbdt
def hyper_opt(libsvmFile):
    datasetSplit(libsvmFile, 'data_train_th100', 'data_test_th100', 0.2, 100000)
    X_train, y_train = load_svmlight_file('data_train_th100')
    X_train = X_train.toarray()
    estimator = hpsklearn.HyperoptEstimator(None,
                                            classifier=hpsklearn.components.any_classifier('clf'),
                                            algo=hyperopt.tpe.suggest,
                                            trial_timeout=10.0,
                                            max_evals=10)
    fit_iterator = estimator.fit_iter(X_train, y_train)
    fit_iterator.next()
    plot_helper = hpsklearn.demo_support.PlotHelper(estimator, mintodate_ylim=(0.0,0.1))
    while len(estimator.trials.trials) < estimator.max_evals:
        fit_iterator.send(1)
        plot_helper.post_iter()
    plot_helper.post_loop()
    estimator.retrain_best_model_on_full_data(X_train, y_train)
    print 'Best classifier: \n', estimator.best_model()

if __name__ == '__main__':
    calcPosNegCnt('label_feature_data_libsvm')
    datasetSplit('50018_20160625_cross_sample', 0.2, 'lr_data_train', 'lr_data_test', 600000)
    xgboost_lr_train_test('data_libsvm_th100')
    lr_train_test('data_libsvm_th500', 'data_cross_libsvm_th500')
    xgb_feature_lr_train_test('xgb_feature_libsvm', 'data_cross_libsvm_th100')
    hyper_opt('data_libsvm_th100')
