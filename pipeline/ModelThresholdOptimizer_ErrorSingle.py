# Select the optimal classification threshold using validation set for LR models
import warnings
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score
from utils import save_json

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def load_model(dir, seed, group_suffix=None, return_val_data=True):
    if group_suffix is not None:
        model = load(dir + '-'.join(['model', str(seed), group_suffix]) + '.joblib')
        Y_val_df = pd.read_csv(dir + '-'.join(['y_val', str(seed), group_suffix]) + '.csv')
    else:
        model = load(dir + '-'.join(['model', str(seed)]) + '.joblib')
        Y_val_df = pd.read_csv(dir + '-'.join(['y_val', str(seed)]) + '.csv')

    if return_val_data:
        return model, Y_val_df
    else:
        return model

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

    return {'thres': best_class_thresh, 'bal_acc': np.max(ba_arr)}

def compute_model_optimal_thresh(data_name, seed, group_suffix, path, opt_obj, special_suffix=None):
    cur_dir = path + data_name + '/'
    if special_suffix is not None:
        group_suffix = group_suffix + '-' + special_suffix
    model, Y_val_df = load_model(cur_dir, seed, group_suffix=group_suffix)
    opt_thres = find_optimal_thres(Y_val_df, opt_obj=opt_obj)

    if group_suffix is not None:
        save_json(opt_thres, cur_dir+'-'.join(['Thres', str(seed), group_suffix]))
    else:
        save_json(opt_thres, cur_dir + '-'.join(['Thres', str(seed)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find optimal threshold on validation set for LR models")

    parser.add_argument("--model", type=str, default='group',
                        help="setting of validation for single or group models, if 'group' run for all the models in MultiCC. Otherwise, only for the single model in SingleCC.")
    parser.add_argument("--setting", type=str, default=None,
                        help="setting of SingleCC. When is not None, called after SingleCC. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively")
    parser.add_argument("--error", type=str,
                        help="setting of error simulation. Choose from [error0.05, error0.10, error0.15, error0.20, error0.25, error0.30] that represent different error rates in the training data respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=1,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()


    # datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    datasets = ['lawgpa']

    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]
    seeds = [6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'
    opt_obj = 'BalAcc'

    if args.model == 'group': # for MultiCC
        settings = [None, 'G0', 'G1']
    elif args.model == 'single': # for SingleCC
        if args.setting is None:
            raise ValueError(
                'The input "setting" is requried. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

        if args.setting not in ['SingleCC', 'KAM-CAL', 'SingleCC+KAM-CAL']:
            raise ValueError(
                'The input "setting" is not supported. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')
        settings = [args.setting]
    else:
        raise ValueError('The input "model" is not supported. Choose from [group, single].')

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


    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds:
                for group_suffix in settings:
                    tasks.append([data_name, seed, group_suffix, res_path, opt_obj, args.error])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(compute_model_optimal_thresh, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')
