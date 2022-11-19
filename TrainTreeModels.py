# Train and store XGBoost tree models
import warnings
import timeit
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from TrainMLModels import Learner, read_json, save_json, generate_model_predictions, find_optimal_thres
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import os
from joblib import dump

warnings.filterwarnings(action='ignore')

class XgBoost(Learner):

    def __init__(self, scoring='accuracy'):
        super(XgBoost, self).__init__(scoring)
        self.name = "XGB"

    def fit(self, train_data, train_Y, cat_cols, seed, sample_weight=None):

        if self.scoring == 'accuracy':
            xg_metric = 'error'

        if self.scoring == 'roc_auc':
            xg_metric = 'auc'

        feature_transformation = ColumnTransformer(transformers=[
            ('encode_categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ])

        param_grid = {
            'learner__n_estimators': [5, 10],
            'learner__max_depth': [2, 3, 5],
            'learner__objective': ['binary:logistic'],
            'learner__eval_metric': [xg_metric],
            'learner__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', xgb.XGBClassifier(random_state=seed))])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=-1, n_jobs=-1)

        if sample_weight is not None:
            try:
                model = search.fit(train_data, train_Y, **{'learner__sample_weight': sample_weight})
            except:
                model = None
        else:
            try:
                model = search.fit(train_data, train_Y)
            except:
                model = None
        return model



def XGB_trainer(data_name, seed, sensi_flag=1, res_path='../intermediate/models/',
                  verbose=True, data_path='data/processed/',
                  y_col='Y', sensi_col='A', n_groups=2):
    start = timeit.default_timer()

    set_suffix = 'S_{}'.format(sensi_flag)
    repo_dir = res_path.replace('intermediate/models/', '')
    cur_dir = res_path + data_name + '/'

    # get binned numerical features
    train_df = pd.read_csv(cur_dir + '-'.join(['train', str(seed), 'bin']) + '.csv')
    val_df = pd.read_csv(cur_dir + '-'.join(['val', str(seed), 'bin']) + '.csv')

    meta_info = read_json(repo_dir + '/' + data_path + data_name + '.json')
    n_features = meta_info['n_features']

    learner = XgBoost()

    if set_suffix == 'S_1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]

    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])
    val_data = val_df[features]


    model = learner.fit(train_data, Y_train, features, seed)
    if model is not None:

        # for optimal threshold selection
        val_df['Y_pred_scores'] = generate_model_predictions(model, val_data)

        dump(model, cur_dir + '-'.join(['trmodel', str(seed), set_suffix]) + '.joblib')

        # optimize threshold first
        opt_thres = find_optimal_thres(val_df, opt_obj='BalAcc')
        par_dict = {'thres': opt_thres['thres'], 'BalAcc': opt_thres['BalAcc']}

        # train group-level models for MultiCC
        for group_i in range(n_groups):
            group_train_df = train_df[train_df[sensi_col] == group_i].copy()
            group_val_df = val_df[val_df[sensi_col] == group_i].copy()

            group_train_data = group_train_df[features]
            group_Y_train = np.array(group_train_df[y_col])

            group_val_data = group_val_df[features]
            group_Y_val = np.array(group_val_df[y_col])

            group_model = learner.fit(group_train_data, group_Y_train, features, seed)

            # for optimal threshold selection
            group_val_df['Y'] = group_Y_val
            group_val_df['Y_pred_scores'] = generate_model_predictions(group_model, group_val_data)

            group_opt_thres = find_optimal_thres(group_val_df, opt_obj='BalAcc')
            par_dict.update({'thres_g{}'.format(group_i): group_opt_thres['thres'],
                             'BalAcc_g{}'.format(group_i): group_opt_thres['BalAcc']})

            dump(group_model, cur_dir + '-'.join(['trmodel', str(seed), 'G' + str(group_i), set_suffix]) + '.joblib')

        end = timeit.default_timer()
        time = end - start
        save_json({'time': time}, '{}trmltime-{}.json'.format(cur_dir, seed))

        save_json(par_dict, '{}trthres-{}-{}.json'.format(cur_dir, seed, set_suffix))

        if verbose:
            Y_train_pred = generate_model_predictions(model, train_data, opt_thres=0.5)
            score_train = learner.score(Y_train, Y_train_pred)
            print('---' * 8, data_name, seed, '---' * 8)
            print(learner.scoring, "on train data: ", score_train)
            print('---' * 10, '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XGBoost Tree models on original data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--sensi", type=int, default=1,
                        help="whether to include the sensitive attribute as a feature in training ML models.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=10,
                        help="number of datasets over which the script is running. Default is 10 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['lsac', 'cardio', 'bank', 'meps16', 'credit', 'ACSE', 'ACSP', 'ACSH', 'ACSM', 'ACSI']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

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
        else:
            datasets = datasets[:n_datasets]

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds:
                tasks.append([data_name, seed, args.sensi, res_path, False])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(XGB_trainer, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')