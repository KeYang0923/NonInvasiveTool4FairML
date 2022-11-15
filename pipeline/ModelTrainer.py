# Train and store Logistic Regression models on original data

import warnings
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from utils import make_folder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from joblib import dump

warnings.filterwarnings(action='ignore')

class Learner(object):
    # code from "https://github.com/schelterlabs/learning-to-validate-predictions/blob/master/pp/learners.py"
    def __init__(self, scoring):
        self.scoring = scoring

    def split(self, data, seed, sizes=[0.7, 0.5]):
        np.random.seed(seed)
        n = data.shape[0]
        split_point = int(sizes[0] * n)
        order = list(np.random.permutation(n))
        train_data = data.iloc[order[:split_point], :]

        vt_data = data.iloc[order[split_point:], :]
        second_n = vt_data.shape[0]
        second_order = list(np.random.permutation(second_n))
        second_split_point = int(sizes[1] * second_n)

        val_data = vt_data.iloc[second_order[:second_split_point], :]
        test_data = vt_data.iloc[second_order[second_split_point:], :]
        return train_data, val_data, test_data

    def scoring_name(self):
        return self.scoring

    def score(self, y_true, y_pred):
        if self.scoring == 'accuracy':
            return accuracy_score(y_true, y_pred)

        if self.scoring == 'roc_auc':
            return roc_auc_score(y_true, y_pred)

        if self.scoring == 'neg_mean_squared_error':
            return mean_squared_error(y_true, y_pred)

        raise Exception('unknown scoring {}'.format(self.scoring))

class SKLogisticRegression(Learner):

    def __init__(self, scoring='accuracy'):
        super(SKLogisticRegression, self).__init__(scoring)
        self.name = "LR"

    def fit(self, train_data, train_Y, transform_cols, sample_weight=None):

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        feature_transformation = ColumnTransformer(transformers=[
            ('scaled_numeric', StandardScaler(), transform_cols),
        ])

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=-1, n_jobs=-1)

        if sample_weight is not None:
            model = search.fit(train_data, train_Y, **{'learner__sample_weight': sample_weight})
        else:
            model = search.fit(train_data, train_Y)

        return model

class AIFLogisticRegression(Learner):
    # For the LR models for benchmark datasets used in AIF 360
    def __init__(self, scoring='accuracy'):
        super(AIFLogisticRegression, self).__init__(scoring)
        self.name = "LR_AIF"

    def fit(self, train_data, train_Y, transform_cols, sample_weight=None):
        feature_transformation = ColumnTransformer(transformers=[
                                ('scaled_numeric', StandardScaler(), transform_cols),
                                ])
        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', LogisticRegression())])
        if sample_weight is not None:
            model = pipeline.fit(train_data, train_Y, **{'learner__sample_weight': sample_weight})
        else:
            model = pipeline.fit(train_data, train_Y)
        return model

def generate_model_predictions(data_name, cur_model, cur_data, opt_thres=None):
    if data_name in ['adult', 'german', 'compas']:
        pos_ind = np.where(cur_model.classes_ == 1.0)[0][0] # default positive label is 1.0
        Y_pred_proba = cur_model.predict_proba(cur_data)[:, pos_ind].reshape(-1, 1)
    elif data_name in ['cardio', 'bank', 'lawgpa', 'meps16', 'credit', 'UFRGS', 'ACS_E', 'ACS_F', 'ACS_P', 'ACS_H', 'ACS_T', 'ACS_M', 'ACS_I']:
        pos_ind = np.where(cur_model.best_estimator_.named_steps['learner'].classes_ == 1.0)[0][0]
        Y_pred_proba = cur_model.predict_proba(cur_data)[:, pos_ind].reshape(-1, 1)
    else:
        raise ValueError('The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS]')
    if opt_thres is not None:
        return [int(y > opt_thres) for y in Y_pred_proba]
    else:
        return Y_pred_proba

def LR_trainer(data_name, y_col, sensi_col, seed, sensi_col_in_training, res_path='../intermediate/models/',
                  verbose=False, n_groups=2, data_path='../data/processed/', file_suffix='_features'):

    cur_dir = res_path + data_name + '/'
    make_folder(cur_dir)
    df = pd.read_csv(data_path + data_name + file_suffix + '.csv')
    # df = pd.read_csv(data_path + data_name + file_suffix + '_opp.csv')

    if data_name in ['adult', 'german', 'compas']:  # reproduce the models used in AIF 360 for benchmark datasets
        learner = AIFLogisticRegression()
    elif data_name in ['cardio', 'bank', 'lawgpa', 'meps16', 'credit', 'UFRGS', 'ACS_E', 'ACS_F', 'ACS_P', 'ACS_H', 'ACS_T', 'ACS_M', 'ACS_I']: # standard logistic regression for other datasets
        learner = SKLogisticRegression()
    else:
        raise ValueError('The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS, ACS_E, ACS_F, ACS_H, ACS_T, ACS_M, ACS_I]')

    train_df, validate_df, test_df = learner.split(df, seed)
    if sensi_col_in_training:
        features = [x for x in train_df.columns if x != y_col]

    else:
        features = [x for x in train_df.columns if x != y_col and x != sensi_col]



    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])


    val_data = validate_df[features]
    Y_val = np.array(validate_df[y_col])

    model = learner.fit(train_data, Y_train, features)

    # for optimal threshold selection
    validate_df['Y'] = Y_val
    validate_df['Y_pred_scores'] = generate_model_predictions(data_name, model, val_data)

    set_suffix = 'S_{}'.format(sensi_col_in_training)

    dump(model, cur_dir + '-'.join(['model', str(seed), set_suffix]) + '.joblib')
    validate_df[['Y', 'Y_pred_scores']].to_csv(cur_dir + '-'.join(['y_val', str(seed), set_suffix]) + '.csv', index=False)

    train_df.to_csv(cur_dir + '-'.join(['train', str(seed), set_suffix]) + '.csv') # keep index for sanity check of random splits
    validate_df.to_csv(cur_dir + '-'.join(['val', str(seed), set_suffix]) + '.csv', index=False)
    test_df.to_csv(cur_dir + '-'.join(['test', str(seed), set_suffix]) + '.csv', index=False)

    # train group-level models for MultiCC
    for group_i in range(n_groups):
        group_train_df = train_df[train_df[sensi_col] == group_i].copy()
        group_val_df = validate_df[validate_df[sensi_col] == group_i].copy()

        group_train_data = group_train_df[features]
        group_Y_train = np.array(group_train_df[y_col])

        group_val_data = group_val_df[features]
        group_Y_val = np.array(group_val_df[y_col])

        group_model = learner.fit(group_train_data, group_Y_train, features)

        # for optimal threshold selection
        group_val_df['Y'] = group_Y_val
        group_val_df['Y_pred_scores'] = generate_model_predictions(data_name, group_model, group_val_data)
        dump(group_model, cur_dir + '-'.join(['model', str(seed), 'G' + str(group_i), set_suffix]) + '.joblib')
        group_val_df[['Y', 'Y_pred_scores']].to_csv(cur_dir + '-'.join(['y_val', str(seed), 'G' + str(group_i), set_suffix]) + '.csv', index=False)

    if verbose:
        Y_train_pred = generate_model_predictions(data_name, model, train_data)
        score_train = learner.score(Y_train, Y_train_pred)
        print('---' * 8, data_name, seed, '---' * 8)
        print(learner.scoring, "on train data: ", score_train)
        print('---' * 10, '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LR models on original data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--sensi", type=int, default=1,
                        help="whether to include the sensitive attribute as a feature in training ML models.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=9,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    # datasets = ['ACS_E', 'ACS_F', 'ACS_P', 'ACS_H', 'ACS_T', 'ACS_M', 'ACS_I']

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS'] #, 'ACS_E', 'ACS_F', 'ACS_P', 'ACS_H', 'ACS_T', 'ACS_M', 'ACS_I']

    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(6)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(6)]

    # datasets = ['UFRGS']
    # y_cols = ['Y']
    # sensi_cols = ['C0']

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    if args.set_n is None:
        raise ValueError(
            'The input "set_n" is requried. Use "--set_n 1" for running over a single dataset.')
    elif type(args.set_n) == str:
        raise ValueError(
            'The input "set_n" requires integer. Use "--set_n 1" for running over a single dataset.')
    else:
        n_datasets = int(args.set_n)
        if n_datasets == -1:
            datasets = datasets[n_datasets:]
            y_cols = y_cols[n_datasets:]
            sensi_cols = sensi_cols[n_datasets:]
        else:
            datasets = datasets[:n_datasets]
            y_cols = y_cols[:n_datasets]
            sensi_cols = sensi_cols[:n_datasets]

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    res_path = '../intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, y_col, sensi_col, seed, args.sensi, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(LR_trainer, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')