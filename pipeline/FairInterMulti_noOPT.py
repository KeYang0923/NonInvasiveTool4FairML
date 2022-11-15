# Use MultiCC and multi-model to produce predictions
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from ModelTrainer import generate_model_predictions
from utils import read_json, load_model

warnings.filterwarnings('ignore')


def assign_pred(x, n_vio): # assign prediction based on minimal violation
    violations = [x.iloc[i] for i in range(n_vio)]
    pred_index = violations.index(min(violations))
    if pred_index <= 1:
        return x.iloc[n_vio]
    else:
        return x.iloc[n_vio+1]
def assign_pred_en(x): # assign prediction based on the group membership of sensitive attribute
    if x.iloc[0]: # sensi_col == 1, majority group
        return x.iloc[2]
    else:
        return x.iloc[1]


def apply_models_by_cc(data_name, seed, y_col, sensi_col, res_path='../intermediate/models/', special_suffix=None,
                       n_groups=2, n_labels=2):
    cur_dir = res_path + data_name + '/'
    test_vio = pd.read_csv(cur_dir + '-'.join(['test_violation', str(seed), special_suffix]) + '.csv')
    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed)]) + '.csv')
    # print(test_vio.columns)

    vio_cols = ['vio_by_' + '_'.join(['noOPT', 'G' + str(group_i), 'L' + str(label_i)]) for group_i in range(n_groups) for label_i in range(n_labels)]
    # features = [x for x in test_df.columns if x!=y_col and x!= sensi_col]
    features = [x for x in test_df.columns if x!=y_col]

    test_data = test_df[features]

    for group_suffix in [None, 'G0', 'G1']:
        model = load_model(cur_dir, seed, group_suffix)
        if group_suffix is not None:
            thres_file = cur_dir + '-'.join(['Thres', str(seed), group_suffix])
            pred_col = 'Y_pred_'+group_suffix
        else:
            thres_file = cur_dir + '-'.join(['Thres', str(seed)])
            pred_col = 'Y_pred'
        opt_thres = read_json(thres_file)['thres']
        test_vio[pred_col] = generate_model_predictions(data_name, model, test_data, opt_thres)

    test_vio['Y_pred_en'] = test_vio[[sensi_col, 'Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_en(x), axis=1)
    test_vio['Y_pred_en_cc'] = test_vio[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred(x, int(n_labels * n_groups)), axis=1)
    test_vio['Y'] = test_vio[y_col]

    test_vio[[sensi_col, 'Y', 'Y_pred', 'Y_pred_G0', 'Y_pred_G1', 'Y_pred_en', 'Y_pred_en_cc']].to_csv(cur_dir + '-'.join(['test_eval_multi', str(seed), special_suffix]) + '.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply MultiCC and multi-model to predict on original test data or erroneous test data under a fixed rate")
    parser.add_argument("--data", type=str,
                        help="name of dataset. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS].")

    parser.add_argument("--special", type=str, default='noOPT',
                        help="setting of density optimization.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=1,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = [args.data]
    if args.data in ['cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']:
        y_cols = ['Y']
        sensi_cols = ['C0']
    elif args.data == 'adult':
        y_cols = ['Income Binary']
        sensi_cols = ['sex']
    elif args.data == 'german':
        y_cols = ['credit']
        sensi_cols = ['age']
    elif args.data == 'compas':
        y_cols = ['two_year_recid']
        sensi_cols = ['race']
    else:
        raise ValueError(
            'The input "data" is not supported. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS] for different datasets.')

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'


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

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, res_path, args.special])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(apply_models_by_cc, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')


