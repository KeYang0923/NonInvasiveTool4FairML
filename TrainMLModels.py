# Train and store Logistic Regression models on original data

import warnings
import timeit
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_squared_error

import os, pathlib, json
from joblib import dump

warnings.filterwarnings(action='ignore')

def make_folder(file_path):
    if not os.path.exists(file_path):
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

def read_json(file_name_with_path):
    if os.path.isfile(file_name_with_path):
        with open(file_name_with_path) as f:
            res = json.load(f)
        return res
    else:
        raise ValueError('Not exist', file_name_with_path)

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

class LogisticRegression(Learner):

    def __init__(self, scoring='accuracy'):
        super(LogisticRegression, self).__init__(scoring)
        self.name = "LR"

    def fit(self, train_data, train_Y, features, seed, sample_weight=None):

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        feature_transformation = ColumnTransformer(transformers=[
            ('scaled_numeric', MinMaxScaler(), features)
        ])

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000, random_state=seed))])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=-1, n_jobs=-1)

        if sample_weight is not None:
            model = search.fit(train_data, train_Y, **{'learner__sample_weight': sample_weight})
        else:
            model = search.fit(train_data, train_Y)

        return model

def generate_model_predictions(cur_model, cur_data, opt_thres=None):
    pos_ind = np.where(cur_model.best_estimator_.named_steps['learner'].classes_ == 1.0)[0][0]
    Y_pred_proba = cur_model.predict_proba(cur_data)[:, pos_ind].reshape(-1, 1)

    if opt_thres is not None:
        return [int(y > opt_thres) for y in Y_pred_proba]
    else:
        return Y_pred_proba

def save_json(input_dict, file_path_with_name, verbose=False):
    with open(file_path_with_name, 'w') as json_file:
        json.dump(input_dict, json_file, indent=2)
    if verbose:
        print('--> Dict is saved to ', file_path_with_name + '\n')

def compute_bal_acc(y_true, y_pred, label_order=[0, 1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=label_order).ravel()
    P = TP + FN
    N = TN + FP
    TPR = TP / P if P > 0.0 else np.float64(0.0)
    TNR = TN / N if N > 0.0 else np.float64(0.0)
    return 0.5 * (TPR + TNR)


def find_optimal_thres(y_val_df, opt_obj='BalAcc', pred_col=None, num_thresh=100, verbose=False):
    if pred_col is None:
        pred_col = 'Y_pred_scores'

    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)

    for idx, class_thresh in enumerate(class_thresh_arr):
        y_val_df['Y_pred_th'] = y_val_df[pred_col].apply(lambda x: x > class_thresh)
        if opt_obj == 'BalAcc':
            ba_arr[idx] = compute_bal_acc(y_val_df['Y'], y_val_df['Y_pred_th'])
        elif opt_obj == 'Acc':
            ba_arr[idx] = accuracy_score(y_val_df['Y'], y_val_df['Y_pred_th'])
        else:
            raise ValueError('The "opt_obj" specified is not supported. Now only support "BalAcc" and "Acc"!')
    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]
    if verbose:
        print("Best balanced accuracy = %.4f" % np.max(ba_arr))
        print("Optimal classification threshold = %.4f" % best_class_thresh)

    return {'thres': best_class_thresh, opt_obj: np.max(ba_arr)}


def LR_trainer(data_name, seed, sensi_col_in_training=True, res_path='../intermediate/models/',
               verbose=True, n_groups=2, data_path='data/processed/', sensi_col = 'A', y_col = 'Y'):
    start = timeit.default_timer()
    cur_dir = res_path + data_name + '/'
    make_folder(cur_dir)

    repo_dir = res_path.replace('intermediate/models/', '')

    df = pd.read_csv(repo_dir+'/'+data_path + data_name + '.csv')
    meta_info = read_json(repo_dir+'/'+data_path + data_name + '.json')

    n_features = meta_info['n_features'] # including sensitive column

    learner = LogisticRegression()

    train_df, validate_df, test_df = learner.split(df, seed)

    # if sensi_col_in_training:
    #     features = ['X{}'.format(i) for i in range(1, n_features)] + ['A']
    # else:
    #     features = ['X{}'.format(i) for i in range(1, n_features)]


    # train_data = train_df[features]
    # Y_train = np.array(train_df[y_col])
    #
    # val_data = validate_df[features]
    # Y_val = np.array(validate_df[y_col])
    #
    # model = learner.fit(train_data, Y_train, features, seed)
    #
    # # for optimal threshold selection
    # validate_df['Y'] = Y_val
    # validate_df['Y_pred_scores'] = generate_model_predictions(model, val_data)
    #
    set_suffix = 'S_{}'.format(sensi_col_in_training)
    #
    # dump(model, cur_dir + '-'.join(['model', str(seed), set_suffix]) + '.joblib')
    # validate_df[['Y', 'Y_pred_scores']].to_csv(cur_dir + '-'.join(['y_val', str(seed), set_suffix]) + '.csv', index=False)

    # train_df.to_csv(cur_dir + '-'.join(['train', str(seed), set_suffix]) + '.csv') # keep index for sanity check of random splits
    # validate_df.to_csv(cur_dir + '-'.join(['val', str(seed), set_suffix]) + '.csv', index=False)
    test_df.to_csv(cur_dir + '-'.join(['test', str(seed), set_suffix]) + '.csv', index=False)

    # optimize threshold first
    # opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
    # par_dict = {'thres': opt_thres['thres'], 'BalAcc': opt_thres['BalAcc']}
    #
    # # train group-level models for MultiCC
    # for group_i in range(n_groups):
    #     group_train_df = train_df[train_df[sensi_col] == group_i].copy()
    #     group_val_df = validate_df[validate_df[sensi_col] == group_i].copy()
    #
    #     group_train_data = group_train_df[features]
    #     group_Y_train = np.array(group_train_df[y_col])
    #
    #     group_val_data = group_val_df[features]
    #     group_Y_val = np.array(group_val_df[y_col])
    #
    #     group_model = learner.fit(group_train_data, group_Y_train, features, seed)
    #
    #     # for optimal threshold selection
    #     group_val_df['Y'] = group_Y_val
    #     group_val_df['Y_pred_scores'] = generate_model_predictions(group_model, group_val_data)
    #
    #     group_opt_thres = find_optimal_thres(group_val_df, opt_obj='BalAcc')
    #     par_dict.update({'thres_g{}'.format(group_i): group_opt_thres['thres'], 'BalAcc_g{}'.format(group_i): group_opt_thres['BalAcc']})
    #
    #     dump(group_model, cur_dir + '-'.join(['model', str(seed), 'G' + str(group_i), set_suffix]) + '.joblib')
    #     # group_val_df[['Y', 'Y_pred_scores']].to_csv(cur_dir + '-'.join(['y_val', str(seed), 'G' + str(group_i), set_suffix]) + '.csv', index=False)
    #
    # end = timeit.default_timer()
    # time = end - start
    # save_json({'time': time}, '{}mltime-{}.json'.format(cur_dir, seed))
    #
    # save_json(par_dict, '{}thres-{}.json'.format(cur_dir, seed))
    #
    # if verbose:
    #     Y_train_pred = generate_model_predictions(model, train_data, opt_thres=0.5)
    #     score_train = learner.score(Y_train, Y_train_pred)
    #     print('---' * 8, data_name, seed, '---' * 8)
    #     print(learner.scoring, "on train data: ", score_train)
    #     print('---' * 10, '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LR models on original data")
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
            pool.starmap(LR_trainer, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')