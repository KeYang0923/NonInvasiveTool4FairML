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


def apply_models_by_cc(data_name, seed, y_col, sensi_col, res_path='../intermediate/models/', error_suffix=None,
                       n_groups=2, n_labels=2):
    cur_dir = res_path + data_name + '/'
    if 'error' in error_suffix: # for erroneous records
        test_vio = pd.read_csv(cur_dir + '-'.join(['test_violation', str(seed), error_suffix]) + '.csv')
        test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed), error_suffix]) + '.csv')
    else: # original data
        test_vio = pd.read_csv(cur_dir + '-'.join(['test_violation', str(seed)]) + '.csv')
        test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed)]) + '.csv')

    vio_cols = ['vio_by_' + '_'.join(['density', 'G' + str(group_i), 'L' + str(label_i)]) for group_i in range(n_groups) for label_i in range(n_labels)]
    features = [x for x in test_df.columns if x!=y_col and x!= sensi_col]

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
    if 'error' in error_suffix: # for erroneous records
        test_vio[[sensi_col, 'Y', 'Y_pred', 'Y_pred_G0', 'Y_pred_G1', 'Y_pred_en', 'Y_pred_en_cc']].to_csv(cur_dir + '-'.join(['test_eval_multi', str(seed), error_suffix]) + '.csv', index=False)
    else:
        test_vio[[sensi_col, 'Y', 'Y_pred', 'Y_pred_G0', 'Y_pred_G1', 'Y_pred_en', 'Y_pred_en_cc']].to_csv(cur_dir + '-'.join(['test_eval_multi', str(seed)]) + '.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply MultiCC and multi-model to predict on original test data or erroneous test data under a fixed rate")
    parser.add_argument("--setting", type=str,
                        help="error rate for MultiCC. When setting is not 'orig', MultiCC and multi-model are running on test set with errors in sensitive attribute. Choose from [error0.2, error0.15, error0.1, error0.05, error0.01] for different rates of errors")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(5)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(5)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'


    if args.setting is None:
        raise ValueError('The input "setting" is requried. Use "drift" for original test without errors OR "error"+0.1 for the example of 10% errors simulated in the test set.')

    elif 'error' not in args.setting:
        if args.setting != 'orig':
            raise ValueError(
                'The input "setting" is not supported. Use "drift" for original test without errors OR "error"+0.1 for the example of 10% errors simulated in the test set.')

    else: # erroneous case
        if args.setting not in ['error0.2', 'error0.15', ' error0.1', 'error0.05', 'error0.01']:
            raise ValueError(
                'The input "setting" is not supported. Use "drift" for original test without errors OR "error"+0.1 for the example of 10% errors simulated in the test set. OR run "ErrorDataSimulator.py" for new desired rates.')

    cc_setting = args.setting

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, res_path, cc_setting])
        with Pool(cpu_count()) as pool:
            pool.starmap(apply_models_by_cc, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')


