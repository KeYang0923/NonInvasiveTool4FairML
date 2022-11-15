# Train and store Logistic Regression models on original data
import warnings
import argparse
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

import json
from TrainMLModels import make_folder, read_json, LogisticRegression

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

def compute_weights(df, method, sample_base='zero', alpha_g0=2.0, alpha_g1=1.0, omn_lam=1.0, cc_col='vio_cc', sensi_col='A', y_col='Y'):

    group_1_y_1 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_1_y_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_0_y_1 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_0_y_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_1 = np.array(df[sensi_col] == 1).astype(int)
    group_0 = np.array(df[sensi_col] == 0).astype(int)
    target_1 = np.array(df[y_col] == 1).astype(int)
    target_0 = np.array(df[y_col] == 0).astype(int)
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

def compute_bal_acc(y_true, y_pred, label_order=[0, 1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=label_order).ravel()
    P = TP + FN
    N = TN + FP
    TPR = TP / P if P > 0.0 else np.float64(0.0)
    TNR = TN / N if N > 0.0 else np.float64(0.0)
    return 0.5 * (TPR + TNR)


def find_optimal_thres(y_val_df, opt_obj='BalAcc', num_thresh=100, verbose=False):
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)

    for idx, class_thresh in enumerate(class_thresh_arr):
        y_val_df['Y_pred'] = y_val_df['Y_pred_scores'].apply(lambda x: x > class_thresh)
        if opt_obj == 'BalAcc':
            ba_arr[idx] = compute_bal_acc(y_val_df['Y'], y_val_df['Y_pred'])
        elif opt_obj == 'Acc':
            ba_arr[idx] = accuracy_score(y_val_df['Y'], y_val_df['Y_pred'])
        else:
            raise ValueError('The "opt_obj" specified is not supported. Now only support "BalAcc" and "Acc"!')
    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]
    if verbose:
        print("Best balanced accuracy = %.4f" % np.max(ba_arr))
        print("Optimal classification threshold = %.4f" % best_class_thresh)

    return {'thres': best_class_thresh, opt_obj: np.max(ba_arr)}


def get_sp(settings, inter_high=10, low=0, res_path='../intermediate/models/',
            set_suffix='S_1', data_path='../data/processed/', y_col = 'Y', sensi_col='A'):
    data_name, seed, reweigh_method, weight_base = settings
    cur_dir = res_path + data_name + '/'
    make_folder(cur_dir)

    train_df = pd.read_csv(cur_dir + '-'.join(['train_vio', str(seed)]) + '.csv')
    validate_df = pd.read_csv(cur_dir + '-'.join(['val', str(seed), set_suffix]) + '.csv')

    meta_info = read_json(data_path + data_name + '.json')
    n_features = meta_info['n_features']  # including sensitive column

    if set_suffix == 'S_1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    val_data = validate_df[features]

    mid = (inter_high + low) / 2

    weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=mid, alpha_g0=inter_high, alpha_g1=mid)
    learner = LogisticRegression()
    model = learner.fit(train_data, Y_train, features, seed, weights)

    validate_df['Y_pred_scores'] = generate_model_predictions(model, val_data)

    # optimize threshold first
    opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
    cur_thresh = opt_thres['thres']

    validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
    cur_sp = eval_sp(validate_df, 'Y_pred')
    cur_acc = compute_bal_acc(validate_df['Y'], validate_df['Y_pred'])
    return cur_sp, cur_acc


def search_inter_high_by_gd(settings, left=0.01, right=10, target=0.1, diff=0.09, num_intervals=10, depth=4, best_so_far=None):
    val_try = (left + right) / 2
    sp_try = get_sp(settings, val_try) or 10
    diff_try = abs(sp_try - target)

    best_so_far = best_so_far or (val_try, sp_try, diff_try)

    if target - diff < sp_try < target + diff:
        best_so_far = (val_try, sp_try, diff_try)
        return best_so_far
    elif right - left < 0.01 or depth == 0:
        return best_so_far
    else:
        interval_size = (right - left) / num_intervals
        cand_vals = np.arange(left, right + interval_size, interval_size)
        sp_vals = []
        for val in cand_vals:
            sp_i = get_sp(settings, val) or 10
            diff_i = abs(sp_i - target)

            if diff_i < best_so_far[2]:
                best_so_far = (val, sp_i, diff_i)
                print(best_so_far)

            if diff_i < diff:
                return best_so_far
            else:
                sp_vals.append(sp_i)

        diff_vals = [abs(sp_val - target) for sp_val in sp_vals]
        idx_max = diff_vals.index(max(diff_vals))
        idx_to_left = max(0, idx_max - 1)
        idx_to_right = min(num_intervals, idx_max + 1)

        answer_left = search_inter_high(settings, cand_vals[0], cand_vals[idx_to_left], target, diff, num_intervals, depth - 1,
                                        best_so_far)
        answer_right = search_inter_high(settings, cand_vals[idx_to_right], cand_vals[-1], target, diff, num_intervals, depth - 1,
                                         best_so_far)
        if answer_left[2] < diff:
            best_so_far = answer_left
            return best_so_far
        elif answer_right[2] < diff:
            best_so_far = answer_right
            return best_so_far

        if answer_left[2] < best_so_far[2]:
            best_so_far = answer_left
            print(best_so_far)

        if answer_right[2] < best_so_far[2]:
            best_so_far = answer_right
            print(best_so_far)

        return best_so_far


def search_inter_high(settings, save_f, val_try, acc_previous, sp_previous, left=0.01, right=2.0, step=0.1):
    val_try = val_try or left
    sp_try, acc_try = get_sp(settings, val_try)
    acc_previous = acc_previous or acc_try
    sp_previous = sp_previous or sp_try

    diff_acc = acc_try - acc_previous

    # best_so_far = best_so_far or (val_try, diff_acc, sp_try)
    best_so_far = (val_try, diff_acc, sp_try)
    format_print('---{} {} {}---'.format(val_try, acc_try, sp_try), save_f)

    if val_try > right or (sp_try > 0 and sp_previous < 0):
        return best_so_far
    else:
        return search_inter_high(settings, save_f, val_try + step, acc_try, sp_try, left, right, step)

def LR_trainer(data_name, seed, reweigh_method, weight_base, verbose, res_path='../intermediate/models/'):
    cur_dir = res_path + data_name + '/'
    settings = (data_name, seed, reweigh_method, weight_base)

    if verbose:
        print_f = open('{}degrees-{}-{}-{}.txt'.format(cur_dir, seed, reweigh_method, weight_base), 'w')
    else:
        print_f = None
    best_high = search_inter_high(settings, print_f, None, None, None)

    if verbose:
        print_f.close()

    print('--->', data_name, reweigh_method, weight_base, 'best high ', best_high)

    # par_dict = {'high': best_high[0], 'SPDiff': best_high[2], 'BalACC': best_high[1]}
    # save_json(par_dict, '{}par-{}-{}-{}.json'.format(cur_dir, seed, reweigh_method, weight_base))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tune Weight Scales on real data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--weight", type=str, default='scc',
                        help="which weighing method to use in training ML models.")
    parser.add_argument("--base", type=str, default='one',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    parser.add_argument("--save", type=int, default=0,
                        help="whether to print the results of degrees into disc.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=8,
                        help="number of datasets over which the script is running. Default is 10 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    # datasets = ['cardio', 'bank', 'meps16', 'lsac', 'credit', 'ACSE', 'ACSP', 'ACSH', 'ACSM', 'ACSI']
    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    # datasets = ['cardio', 'bank', 'meps16', 'lsac', 'ACSE', 'ACSP', 'ACSM', 'ACSI']
    # seeds = [1, 12345, 6, 2211, 15]

    datasets = ['lsac']
    seeds = [1]

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

    res_path = '../intermediate/models/'
    for data_name in datasets:
        for seed in seeds:
            LR_trainer(data_name, seed, args.weight, args.base, args.save, res_path)

    # if args.run == 'parallel':
    #     tasks = []
    #     for data_name in datasets:
    #         for seed in seeds:
    #             tasks.append([data_name, seed, args.weight, args.base, args.depth, res_path])
    #     with Pool(cpu_count()//2) as pool:
    #         pool.starmap(LR_trainer, tasks)
    # else:
    #     raise ValueError('Do not support serial execution. Use "--run parallel"!')