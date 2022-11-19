# Train and store Logistic Regression models on original data
import warnings
import argparse, os
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

import json
from TrainMLModels import make_folder, read_json, LogisticRegression, generate_model_predictions, find_optimal_thres, compute_bal_acc

warnings.filterwarnings(action='ignore')

def format_print(str_input, output_f=None):
    if output_f is not None:
        print(str_input, file=output_f)
    else:
        print(str_input)

def compute_sr(y_true, y_pred, label_order=[0, 1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=label_order).ravel()
    pred_P = TP+FP
    P = TP + FN
    N = TN + FP
    return pred_P / (P+N)

def eval_sp(test_eval_df, pred_col, sensi_col='A', n_groups=2):
    SR_all = []
    for group_i in range(n_groups):
        group_df = test_eval_df[test_eval_df[sensi_col] == group_i]
        group_sr = compute_sr(group_df['Y'], group_df[pred_col])
        SR_all.append(group_sr)

    sp_diff = SR_all[0] - SR_all[1]
    return sp_diff

def compute_weights(df, method, sample_base='zero', alpha_g0=2.0, alpha_g1=1.0, omn_lam=1.0, cc_col='vio_cc', sensi_col='A', y_col='Y'):

    group_1_y_1 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_1_y_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_0_y_1 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_0_y_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_1 = np.array(df[sensi_col] == 1).astype(int)
    group_0 = np.array(df[sensi_col] == 0).astype(int)
    target_1 = np.array(df[y_col] == 1).astype(int)
    target_0 = np.array(df[y_col] == 0).astype(int)
    if method == 'scc':
        group_1_y_1_vio_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 1).astype(int) * np.array(df[cc_col] == 0).astype(int)
        group_1_y_0_vio_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 0).astype(int) * np.array(df[cc_col] == 0).astype(int)
        group_0_y_1_vio_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 1).astype(int) * np.array(df[cc_col] == 0).astype(int)
        group_0_y_0_vio_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 0).astype(int) * np.array(df[cc_col] == 0).astype(int)

    total_n = df.shape[0]
    if method == 'scc':
        if sample_base == 'kam':
            sample_weights = np.zeros(total_n)

            sample_weights += group_1_y_1 * (np.sum(group_1) * np.sum(target_1) / (total_n * np.sum(group_1_y_1))) \
                              + group_1_y_0 * (np.sum(group_1) * np.sum(target_0) / (total_n * np.sum(group_1_y_0))) \
                              + group_0_y_1 * (np.sum(group_0) * np.sum(target_1) / (total_n * np.sum(group_0_y_1))) \
                              + group_0_y_0 * (np.sum(group_0) * np.sum(target_0) / (total_n * np.sum(group_0_y_0)))
        elif sample_base == 'omn':
            sample_weights = np.ones(total_n)
            sample_weights -= omn_lam * total_n / np.sum(group_1) * group_1_y_1 \
                              - omn_lam * total_n / np.sum(group_1) * group_1_y_0 \
                              - omn_lam * total_n / np.sum(group_0) * group_0_y_1 \
                              + omn_lam * total_n / np.sum(group_0) * group_0_y_0

        elif sample_base == 'zero':
            sample_weights = np.zeros(total_n)
        elif sample_base == 'one':
            sample_weights = np.ones(total_n)
        else:
            raise ValueError('The input sample_base parameter is not supported. Choose from "[kam, omn, zero, one]".')

        sample_weights -= alpha_g1 * group_1_y_1_vio_0 \
                          - alpha_g1 * group_1_y_0_vio_0 \
                          - alpha_g0 * group_0_y_1_vio_0 \
                          + alpha_g0 * group_0_y_0_vio_0
    elif method == 'kam':
        sample_weights = np.zeros(total_n)

        sample_weights += group_1_y_1 * (np.sum(group_1) * np.sum(target_1) / (total_n * np.sum(group_1_y_1))) \
                          + group_1_y_0 * (np.sum(group_1) * np.sum(target_0) / (total_n * np.sum(group_1_y_0))) \
                          + group_0_y_1 * (np.sum(group_0) * np.sum(target_1) / (total_n * np.sum(group_0_y_1))) \
                          + group_0_y_0 * (np.sum(group_0) * np.sum(target_0) / (total_n * np.sum(group_0_y_0)))
    elif method == 'omn':
        sample_weights = np.ones(total_n)
        sample_weights -= omn_lam * total_n / np.sum(group_1) * group_1_y_1 \
                          - omn_lam * total_n / np.sum(group_1) * group_1_y_0 \
                          - omn_lam * total_n / np.sum(group_0) * group_0_y_1 \
                          + omn_lam * total_n / np.sum(group_0) * group_0_y_0

    else:
        raise ValueError('The input method parameter is not supported. Choose from "[kam, omn, scc]".')

    return sample_weights


def retrain_LR_all_degrees(data_name, seed, reweigh_method, weight_base, verbose, res_path='../intermediate/models/',
               degree_start=0.01, degree_end=2.0, degree_step=0.01,
               set_suffix='S_1', data_path='data/processed/', y_col = 'Y', sensi_col='A'):
    cur_dir = res_path + data_name + '/'
    repo_dir = res_path.replace('intermediate/models/', '')

    train_df = pd.read_csv(cur_dir + '-'.join(['train_vio', str(seed)]) + '.csv')
    validate_df = pd.read_csv(cur_dir + '-'.join(['val', str(seed), set_suffix]) + '.csv')

    meta_info = read_json(repo_dir + data_path + data_name + '.json')
    n_features = meta_info['n_features']  # including sensitive column

    if set_suffix == 'S_1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])
    val_data = validate_df[features]

    if verbose: # save the experimental intervention degree on disc
        print_f = open('{}degrees-{}-{}-{}.txt'.format(cur_dir, seed, reweigh_method, weight_base), 'w')
    else: # print out the experimental intervention degree
        print_f = None

    cur_degree = degree_start
    while cur_degree < degree_end:
        weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=cur_degree, alpha_g0=cur_degree, alpha_g1=cur_degree/2)
        learner = LogisticRegression()
        model = learner.fit(train_data, Y_train, features, seed, weights)

        validate_df['Y_pred_scores'] = generate_model_predictions(model, val_data)

        # optimize threshold first
        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
        cur_thresh = opt_thres['thres']

        validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
        cur_sp = eval_sp(validate_df, 'Y_pred')
        cur_acc = compute_bal_acc(validate_df['Y'], validate_df['Y_pred'])

        format_print('---{} {} {}---'.format(cur_degree, cur_acc, cur_sp), print_f)

        cur_degree += degree_step
    if verbose:
        print_f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Iterate all intervention degrees on real data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--weight", type=str, default='scc',
                        help="which weighing method to use in training ML models.")
    parser.add_argument("--base", type=str, default='one',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    parser.add_argument("--save", type=int, default=1,
                        help="whether to print the results of degrees into disc.")
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
                tasks.append([data_name, seed, args.weight, args.base, args.save, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(retrain_LR_all_degrees, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')