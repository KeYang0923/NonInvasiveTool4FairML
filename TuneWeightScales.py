# Train and store Logistic Regression models on original data
import warnings
import argparse, os
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from PrepareData import read_json
from TrainMLModels import LogisticRegression, XgBoost, generate_model_predictions, find_optimal_thres, compute_bal_acc

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

def compute_weights(df, method, sample_base='zero', alpha_g0=2.0, alpha_g1=1.0, omn_lam=1.0,
                    cc_par=None, cc_col='vio_cc', cc_vio_thres=0.1, cc_reverse=False, sensi_col='A', y_col='Y'):

    group_1_y_1 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_1_y_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_0_y_1 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_0_y_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_1 = np.array(df[sensi_col] == 1).astype(int)
    group_0 = np.array(df[sensi_col] == 0).astype(int)
    target_1 = np.array(df[y_col] == 1).astype(int)
    target_0 = np.array(df[y_col] == 0).astype(int)
    if method == 'scc':
        if cc_par is not None: # if mean(violation) > 0.1, use the corresponding zero violations
            group_1_y_1_mean = int(cc_par['mean_train_G1_L1'] >= cc_vio_thres)
            group_1_y_0_mean = int(cc_par['mean_train_G1_L0'] >= cc_vio_thres)
            group_0_y_1_mean = int(cc_par['mean_train_G0_L1'] >= cc_vio_thres)
            group_0_y_0_mean = int(cc_par['mean_train_G0_L0'] >= cc_vio_thres)
        else:
            group_1_y_1_mean = 1
            group_1_y_0_mean = 1
            group_0_y_1_mean = 1
            group_0_y_0_mean = 1

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
        if cc_reverse:
            sample_weights += alpha_g1 * group_1_y_1_mean * group_1_y_1_vio_0 \
                              - alpha_g1 * group_1_y_0_mean * group_1_y_0_vio_0 \
                              - alpha_g0 * group_0_y_1_mean * group_0_y_1_vio_0 \
                              + alpha_g0 * group_0_y_0_mean * group_0_y_0_vio_0
        else:
            sample_weights -= alpha_g1 * group_1_y_1_mean * group_1_y_1_vio_0 \
                              - alpha_g1 * group_1_y_0_mean * group_1_y_0_vio_0 \
                              - alpha_g0 * group_0_y_1_mean * group_0_y_1_vio_0 \
                              + alpha_g0 * group_0_y_0_mean * group_0_y_0_vio_0
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


def retrain_ML_models_all_degrees(data_name, seed, model_name, reweigh_method, weight_base, verbose, res_path='../intermediate/models/',
               degree_start=0.01, degree_end=2.0, degree_step=0.01,
               data_path='data/processed/', y_col = 'Y', sensi_col='A'):
    cur_dir = res_path + data_name + '/'
    repo_dir = res_path.replace('intermediate/models/', '')

    if model_name == 'tr':
        train_df = pd.read_csv('{}train-{}-bin.csv'.format(cur_dir, seed))
        val_df = pd.read_csv('{}val-{}-bin.csv'.format(cur_dir, seed))

        vio_train_df = pd.read_csv('{}train-cc-{}.csv'.format(cur_dir, seed))
        train_df['vio_cc'] = vio_train_df['vio_cc']

    else:
        train_df = pd.read_csv('{}train-cc-{}.csv'.format(cur_dir, seed))
        val_df = pd.read_csv('{}val-cc-{}.csv'.format(cur_dir, seed))

    meta_info = read_json('{}/{}{}.json'.format(repo_dir, data_path, data_name))

    cc_par = read_json('{}par-cc-{}.json'.format(cur_dir, seed))
    feature_setting = read_json('{}par-{}-{}.json'.format(cur_dir, model_name, seed))['model_setting']

    n_features = meta_info['n_features']  # including sensitive column

    if feature_setting == 'S1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]

    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])
    val_data = val_df[features]

    if verbose: # save the experimental intervention degree on disc
        print_f = open('{}degrees-{}-{}-{}-{}.txt'.format(cur_dir, model_name, seed, reweigh_method, weight_base), 'w')
    else: # print out the experimental intervention degree
        print_f = None

    cur_degree = degree_start
    reverse_flag = False
    while cur_degree < degree_end:
        weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=cur_degree,
                                  alpha_g0=cur_degree, alpha_g1=cur_degree/2, cc_par=cc_par, cc_reverse=reverse_flag)
        if model_name == 'tr':
            learner = XgBoost()
        else:
            learner = LogisticRegression()
        model = learner.fit(train_data, Y_train, features, seed, weights)
        if model is not None:
            val_df['Y_pred_scores'] = generate_model_predictions(model, val_data)

            opt_thres = find_optimal_thres(val_df, opt_obj='BalAcc')
            cur_thresh = opt_thres['thres']

            val_df['Y_pred'] = val_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
            cur_sp = eval_sp(val_df, 'Y_pred')
            cur_acc = compute_bal_acc(val_df['Y'], val_df['Y_pred'])

            format_print('---{} {} {} {}---'.format(model_name, cur_degree, cur_acc, cur_sp), print_f)
            if cur_sp > 0 and reweigh_method == 'scc' and weight_base == 'one':
                reverse_flag = True
        else:
            print('no model fitted ', data_name, model_name, seed, reweigh_method, weight_base)
        cur_degree += degree_step
    if verbose:
        print_f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Iterate all intervention degrees on real data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")

    parser.add_argument("--weight", type=str, default='scc',
                        help="which weighing method to use in training ML models.")
    parser.add_argument("--base", type=str, default='one',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    parser.add_argument("--save", type=int, default=1,
                        help="whether to print the results of degrees into disc.")

    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac'] #, 'bank', 'cardio', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']
    # datasets = ['ACSP', 'ACSH']

    seeds = [1, 12345, 6, 2211, 15] #, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]
    models = ['lr', 'tr']

    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    else:
        raise ValueError(
            'The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

    if args.set_n is not None:
        if type(args.set_n) == str:
            raise ValueError(
                'The input "set_n" requires integer. Use "--set_n 1" for running over a single dataset.')
        else:
            n_datasets = int(args.set_n)
            if n_datasets < 0:
                datasets = datasets[n_datasets:]
            elif n_datasets > 0:
                datasets = datasets[:n_datasets]
            else:
                raise ValueError(
                    'The input "set_n" requires non-zero integer. Use "--set_n 1" for running over a single dataset.')
    if args.model == 'all':
        pass
    elif args.model in models:
        models = [args.model]
    else:
        raise ValueError('The input "model" is not valid. CHOOSE FROM ["lr", "tr"].')

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
                for model_i in models:
                    tasks.append([data_name, seed, model_i, args.weight, args.base, args.save, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(retrain_ML_models_all_degrees, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')