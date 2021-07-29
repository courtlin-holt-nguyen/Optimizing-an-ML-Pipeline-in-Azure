# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, I built and optimized an Azure ML pipeline using the Python SDK, a provided Scikit-learn logistic regression model and Azure HyperDrive.
This HyperDrive optimized model was then compared to an Azure AutoML run to determine which method produced the most accurate model.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
The dataset for this project can be found on the UCI Machine Learning repository. The data is a collection of data points related to the direct marketing efforts of a Portugese bank as it attempted to sell a product, in this case a term deposit, to its customers over the phone. 
We seek to predict whether a customer accepted the bank's product offer or not (variable y). This is a classification task. 

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performing model was a VotingEnsemble model produced by Azure AutoML with an accuracy of: 91.772% from run Id: AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_103

The accuracy of the SciKit-Learn logistic regression model was very similar, with an accuracy of: 91.183% from run Id: HD_a832a5c4-7c1c-4ac0-b1fd-f3c81ecc34e3_0 with the following metrics: Best Metrics: {'Regularization Strength:': 1.0, 'Max iterations:': 20, 'accuracy': 0.9118361153262519}
  

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

PIPELINE ARCHITECTURE
The pipeline architecture used for this project included a SciKit-Learn model optimized with HyperDrive and an AutoML model that automatically generated and evaluated various classification models.

The SciKit-Learn Logistic Regression model was defined in a train.py file and then fed to the HyperDrive with a set of discrete hyperparameters to run. Within the train.py file, TabularDatasetFactory was used to import the banking data, then a custom data cleaning function was called on the data before splitting it into train and test sets and sending it to the Logistic Regression model. HyperDrive attempted to optimize the model by using various values for C (the inverse of the regularization strength) and max_iter (the maximum number of iterations to test). 
The python SDK inside a Notebook file was used to provision a ComputerCluster, configure and run HyperDrive to test various C and max_iter combinations and then save the best model that was found. RandomParameterSampling and a Bandit early terminiation policy were used to reduce the optimization time and cost.

The best Logistic Regression model had the following parameters:
Best Metrics: {'Regularization Strength:': 1.0, 'Max iterations:': 20, 'accuracy': 0.9118361153262519}

The output of the HyperDrive model was then compared to the Accuracy of the Azure AutoML-generated model, which was a VotingEnsemble.  Using the Notebook file, Azure TabularDatasetFactory was used to import the banking data, call the custom clean_data function from the train.py file, configure and run AutoML and then save the best model found.

The final 'VotingEnsemble' consisted of:
 run_algorithm': 'VotingEnsemble',
  'ensembled_iterations': '[88, 69, 94, 30, 39, 97, 53, 57]',
  'ensembled_algorithms': "['XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier']",
  'ensembled_run_ids': "['AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_88', 'AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_69', 'AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_94', 'AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_30', 'AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_39', 'AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_97', 'AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_53', 'AutoML_c4580b80-2401-4a12-8df3-0dccd3a4570d_57']",
  'ensemble_weights': '[0.2727272727272727, 0.09090909090909091, 0.18181818181818182, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091]
 
DATASET
There are 17 attributes in the dataset, as described by UCI:

Input variables:

Bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')

Related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

Other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

Social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

**What are the benefits of the parameter sampler you chose?**
The benefits of the RandomParameterSampling sampler compared to other methods such as GridSearch, is that Random Sampling is less computationally intense, which saves money and time, while still providing very good tuned results. With this method, hyperparameter values are randomly chosen from the defined search space and low-performing runs can be terminated early to further save time and money.

**What are the benefits of the early stopping policy you chose?**
Early stopping policies terminate poorly performing runs to improve computational efficiency.  The benefit of the Bandit early terminiation policy is that it will end runs when the primary metric falls outside of a specified range of the most successful run. 

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
The AutoML method chose a VotingEnsemble comprised of 8 Classifier models (7 XGBoostClassifier models and 1 LightGMB Classifier model) with various weights. The specific models and their hyperparameters were:

prefittedsoftvotingclassifier
{'estimators': ['88', '69', '94', '30', '39', '97', '53', '57'],
 'weights': [0.2727272727272727,
             0.09090909090909091,
             0.18181818181818182,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091,
             0.09090909090909091]}

88 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

88 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.8,
 'eta': 0.3,
 'gamma': 5,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 63,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 2,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.7708333333333335,
 'reg_lambda': 0.10416666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.8,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

69 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

69 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'colsample_bytree': 0.8911111111111111,
 'learning_rate': 0.05263631578947369,
 'max_bin': 160,
 'max_depth': 10,
 'min_child_weight': 4,
 'min_data_in_leaf': 1e-05,
 'min_split_gain': 0.3684210526315789,
 'n_estimators': 800,
 'n_jobs': 2,
 'num_leaves': 107,
 'problem_info': ProblemInfo(
    dataset_samples=32950,
    dataset_features=132,
    dataset_classes=None,
    dataset_num_categorical=0,
    dataset_categoricals=None,
    pipeline_categoricals=None,
    dataset_y_std=None,
    dataset_uid=None,
    subsampling=False,
    task='classification',
    metric=None,
    num_threads=2,
    pipeline_profile='none',
    is_sparse=True,
    runtime_constraints={'mem_in_mb': None, 'wall_time_in_s': None, 'total_wall_time_in_s': 31449600, 'cpu_time_in_s': None, 'num_processes': None, 'grace_period_in_s': None},
    constraint_mode=1,
    cost_mode=1,
    training_percent=100,
    num_recommendations=1,
    model_names_whitelisted=None,
    model_names_blacklisted=None,
    kernel='linear',
    subsampling_treatment='linear',
    subsampling_schedule='hyperband_clip',
    cost_mode_param=None,
    iteration_timeout_mode=0,
    iteration_timeout_param=None,
    feature_column_names=None,
    label_column_name=None,
    weight_column_name=None,
    cv_split_column_names=None,
    enable_streaming=None,
    timeseries_param_dict=None,
    gpu_training_param_dict={'processing_unit_type': 'cpu'}
),
 'random_state': None,
 'reg_alpha': 0.894736842105263,
 'reg_lambda': 0.894736842105263,
 'subsample': 0.99}

94 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

94 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'eta': 0.01,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 7,
 'max_leaves': 15,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 50,
 'n_jobs': 2,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.5625,
 'reg_lambda': 1.9791666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.9,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

30 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

30 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.8,
 'eta': 0.3,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 31,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 2,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 2.5,
 'reg_lambda': 1.0416666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.8,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

39 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

39 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.7,
 'eta': 0.05,
 'gamma': 1,
 'grow_policy': 'lossguide',
 'learning_rate': 0.1,
 'max_bin': 63,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 0,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 2,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.5625,
 'reg_lambda': 1.0416666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.7,
 'tree_method': 'hist',
 'verbose': -10,
 'verbosity': 0}

97 - sparsenormalizer
{'copy': True, 'norm': 'l2'}

97 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.8,
 'eta': 0.3,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 4,
 'max_leaves': 15,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 2,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1.1458333333333335,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.7,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

53 - sparsenormalizer
{'copy': True, 'norm': 'max'}

53 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 0.7,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'eta': 0.3,
 'gamma': 0,
 'grow_policy': 'lossguide',
 'learning_rate': 0.1,
 'max_bin': 255,
 'max_delta_step': 0,
 'max_depth': 9,
 'max_leaves': 63,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 25,
 'n_jobs': 2,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0.8333333333333334,
 'reg_lambda': 0.625,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'tree_method': 'hist',
 'verbose': -10,
 'verbosity': 0}

57 - sparsenormalizer
{'copy': True, 'norm': 'l2'}

57 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'eta': 0.001,
 'gamma': 5,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 7,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 200,
 'n_jobs': 2,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.4583333333333335,
 'reg_lambda': 0.625,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.6,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0} 

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
There was only a slight difference in Accuracy between the Hyperdrive/SciKit-Learn model and the AutoML model. The AutoML model's accuracy was 91.754% while the HyperDrive/SciKit-Learn model's Accuracy was 91.183%
The SciKit-Learn model used a specified model, Logistic Regression, and a set of defined hyperparameters to test. In contrast, the AutoML method tried nearly 100 different classification models with various hyperparameters as well as different ensembles of models to find the optimal one.
 
The difference with the SciKit-Learn method was that it required the user to define the model and hyperparameters, which requires a certain level of expertise with classification problems. On the other hand, the AutoML method would allow someone with very little knowledge of classification problems to test a wide variety of models and ensembles of models. However, the time and computation expense required to run the AutoML model was significantly greater than the Hyperdrive/SciKit-Learn method for only a small improvement in accuracy.   

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
One area for improvement would be to address the class imbalance within the dataset before training the models. As part of the AutoML process, the dataset was checked for class imbalance, which occurs when the number of samples in the minority class (the YES results in this case) are less than 20% of the total samples. AutoML detected that only 11% of the samples in the dataset were Yes results. Training the model with such an imbalanced dataset could lead the resulting model to be perceived as more accurate than it really is. To address class imbalance, the model could be optimized for a different primary metric, such as AUC_weighted. 

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
The cluster was automatically deleted at the end of the experiment using the following code:
cpu_cluster.delete()

And the status was confirmed to be deleting using the code:
cpu_cluster.get_status()
