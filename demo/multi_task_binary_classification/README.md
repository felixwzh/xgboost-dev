# Multi-task Binary Classification

This is the quick start tutorial for MT-GBDT. Here we demonstrate how to use MT-GBDT for a multi-task binary classifcation task. Before getting started, make sure you compile MT-GBDT in the root directory of the project by typing ```make```, and then install python package by ```cd python-package``` and typing ```sudo python setup.py install```. The python file named with "basic_walkthrough.py" can be used to run the demo directly. Here we use [amazon dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) taken from Amazon.com by JHU.

### Tutorial

#### Input Data

Multi-task GBDT based on XGBoost takes  LibSVM format. An example of faked input data is below:

```
1 0:1 1:6.0 2:8.0
0 0:3 1:1.0 2:2.0
...
```

Each line represent a single instance, and in the first line '1' is the instance label, '0', '1', and '2' are the feature indices. In the multi-task binary classification case, instance label '1' is used to indicate positive samples, and '0' is used to indicate negative samples. The first dimension of feature with index '0:' indicates the task id, here the '0:1' and '0:3' means that the first line represents instance from task 1 while the second line represents instance from task 3. '6.0', '8.0', '1.0', and '2.0' are feature values.

We have already transform the raw Amazon dataset into classic LibSVM format and split the data into training set and validation set. The two files, 'amazon.train.data' and 'amazon.val.data' containing instances from all 4 tasks, while 'amazon.val_1.data', 'amazon.val_2.data', 'amazon.val_3.data' and 'amazon.val_4.data' containing instances from single task are subsets of 'amazon.val.data'.

#### Training

Then we can run the training process:

```
python basic_walkthrough.py
```

The walkthrough python file use the dictionary of parameters as below, each item containing the [attribute]:[value] configuration.

```
param = {
        # parameters to run MT-GBDT, not need to tune for using MT-GBDT
        # this is used to indicate which value is used to perform OLF split
        'which_task_value': 2,
        # whether to use task gain self, otherwise, use task gain all
        'use_task_gain_self': 0,
        # which kind of condition we will use to determine when to do task split
        'when_task_split': 1,
        # which kind of task split way we will use to do task split
        'how_task_split': 0,
        
        # parameters for specific tasks, need to tune when dealing with different tasks
        # the max negative sample ratio to determine when to do task split, i.e. the 'threshold ratio R' in paper
        'max_neg_sample_ratio': 0.5,
        # specific tasks related parameters, for Amazon dataset here
        'tasks_list_': (1, 2, 3, 4),
        'task_num_for_init_vec': 5,
        'task_num_for_OLF': 4,
        
        # Tree booster parameters
        # learning step size for a time
        'learning_rate': 0.3,
        # maximum depth of a tree
        'max_depth': 9,
        # minimum amount of hessian(weight) allowed in a child
        'min_child_weight': 1,
        # L2 regularization factor
        'reg_lambda': 12,
        # L1 regularization factor
        ‘reg_alpha’: 0.0005,
        # whether we want to do subsample
        'subsample': 1,
        # whether to subsample columns each split, in each level
        'colsample_bylevel': 0.8,
        # whether to subsample columns during tree construction
        'colsample_bytree': 1.0,
        # minimum loss function reduction required for node splitting
        'gamma': 0.45,
        
        # General parameters
        # whether to not print info during training
        'silent': 1,
        'n_estimators': 1000,
        'early_stopping_rounds': 100,
        # When class labels are unbalanced, setting this parameter positive to make the algorithm converge faster
        'scale_pos_weight': 1,
        'nthread': 8,
        'seed': args.seed,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'exact',
    }
```

In this example, we set 'taks_list_': (1, 2, 3, 4), 'task_num_for_init_vec': 5, and 'task_num_for_OLF': 4 for Amazon multi-task binary classification dataset since it containing 4 different tasks with task id '1', '2', '3', and '4'.

#### Monitoring Progress

When you run training we can find there are messages displayed on screen

```
[19:21:31] src/tree/updater_colmaker.cc:358:   4  task split nodes, 8 pruned task nodes
[0]	train-auc:0.811478	eval-auc:0.777517

[19:21:37] src/tree/updater_colmaker.cc:358:   2  task split nodes, 1 pruned task nodes
[1]	train-auc:0.831786	eval-auc:0.797907
```

If you want only to log the evaluation progress, simply type

```
python basic_walkthrough.py > log.txt
```

You can both monitor progress on screen and record them to log with ```tee```

```
python basic_walkthrough.py | tee log.txt
```

The final prediction results with training model are displayed after training process is finished

```
task 1 's AUC=0.8179795 logloss=0.5378091118242592 at 9 tree
task 2 's AUC=0.8438275 logloss=0.5148902095705271 at 9 tree
task 3 's AUC=0.8892865 logloss=0.4618236219417304 at 9 tree
task 4 's AUC=0.9113155 logloss=0.4355072839874774 at 9 tree
```

#### Training Results

Save and load the model and data with ```pickle```

```
# save model
with open('mt-gbdt.model', 'wb') as model:
	pickle.dump(bst, model)
	
# load model
with open('mt-gbdt.model', 'rb') as model:
	bst = pickle.load(model)
```

The validation results of current model and corresponding parameters are saved to file 'result.csv'.