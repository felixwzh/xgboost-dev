#!/usr/bin/python
# -*- coding=utf-8 -*-

import xgboost as xgb
import random
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc
import numpy as np
import argparse
import os
import pickle


def mkdir(path):
    path = path.strip()
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default='./')  # folder that stores the log
    parser.add_argument('-s', '--seed', type=int, default=27)

    args = parser.parse_args()

    random.seed(args.seed)

    param = {
        "early_stopping_rounds": 100,
        "reg_alpha": 0.0005,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 0.8,
        "scale_pos_weight": 1,
        "learning_rate": 0.3,
        "nthread": 8,
        "min_child_weight": 1,
        "n_estimators": 1000,
        "subsample": 1,
        "reg_lambda": 12,
        "seed": args.seed,
        "objective": 'binary:logistic',
        "max_depth": 9,
        "gamma": 0.45,
        'eval_metric': 'auc',
        'silent': 1,
        'tree_method': 'exact',
        'debug': 0,
        'use_task_gain_self': 0,
        'when_task_split': 1,
        'how_task_split': 0,
        'min_task_gain': 0.0,
        'task_gain_margin': 0.0,
        'max_neg_sample_ratio': 0.5,
        'which_task_value': 2,
        'baseline_alpha': 1.0,
        'baseline_lambda': 1.0,
        'tasks_list_': (1, 2, 3, 4),
        'task_num_for_init_vec': 5,
        'task_num_for_OLF': 4,
    }

    folder = args.folder

    mkdir(folder)
    data_folder = './amazon.'

    # load data
    dtrain = xgb.DMatrix(data_folder + 'train.data')
    dtest = xgb.DMatrix(data_folder + 'val.data')
    deval = xgb.DMatrix(data_folder + 'val.data')

    fout = open(folder+'result.csv', 'a')

    vals = [None] * 30
    for task in param['tasks_list_']:
        vals[task] = xgb.DMatrix(data_folder + 'val_' + str(task) + '.data')

    # train
    evallist = [(dtrain, 'train'), (deval, 'eval')]
    bst = xgb.train(param, dtrain, param['n_estimators'], early_stopping_rounds=param['early_stopping_rounds'], evals=evallist)
    y_real = dtest.get_label()
    y_score = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    # save model
    # with open('mt-gbdt.model', 'wb') as model:
    #     pickle.dump(bst, model)
    # load model
    # with open('mt-gbdt.model', 'rb') as model:
    #     bst = pickle.load(model)

    # compute ROC
    fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
    all_roc_auc = auc(fpr, tpr)
    all_logloss = log_loss(y_real, y_score)

    # output
    fout.write('\n')
    for key in param:
        fout.write(str(key))
        fout.write(',{},'.format(param[key]))
    fout.write('\n')
    fout.write('task,auc,\n')
    for task in param['tasks_list_']:
        best_auc = 0.5
        best_logloss = 0
        y_real = vals[task].get_label()
        tree_num = 0
        for tree in range(2, bst.best_ntree_limit):
            y_score = bst.predict(vals[task], ntree_limit=tree)
            fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
            roc_auc = auc(fpr, tpr)
            logloss = log_loss(y_real, y_score)
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_logloss = logloss
                tree_num = tree
        # acc = accuracy_score(y_real, y_score)

        print("task {} 's AUC={} logloss={} at {} tree".format(task, best_auc, best_logloss, tree_num))
        fout.write("{},{},{}\n".format(task, best_auc, best_logloss))
    fout.write("all,{},{},\n".format(all_roc_auc,all_logloss))

    fout.close()
