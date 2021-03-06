# Evaluate SingleCC, KAM-CAL, SingleCC+KAM-CAL over original test data
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from joblib import load
from utils import read_json, save_json, eval_settings
from ModelTrainer import generate_model_predictions
warnings.filterwarnings('ignore')


def load_model_lr(dir, seed, group_suffix=None):
    if group_suffix is not None:
        model = load(dir + '-'.join(['model', str(seed), group_suffix]) + '.joblib')
    else:
        model = load(dir + '-'.join(['model', str(seed)]) + '.joblib')
    return model


def eval_single_models(data_name, seed, y_col, sensi_col, fair_setting, same_thres=False, res_path='../intermediate/models/'):
    cur_dir = res_path + data_name + '/'

    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed)]) + '.csv')

    features = [x for x in test_df.columns if x != y_col and x != sensi_col]
    orig_model = load_model_lr(cur_dir, seed)
    orig_thres = read_json(cur_dir + '-'.join(['Thres', str(seed)]))['thres']

    cur_model = load_model_lr(cur_dir, seed, fair_setting)
    if same_thres:
        cur_thres = orig_thres
    else:
        cur_thres = read_json(cur_dir + '-'.join(['Thres', str(seed), fair_setting]))['thres']
    test_data = test_df[features]
    test_df['Y_pred_cc'] = generate_model_predictions(data_name, cur_model, test_data, cur_thres)
    test_df['Y_pred'] = generate_model_predictions(data_name, orig_model, test_data, orig_thres)
    test_df['Y'] = test_df[y_col]
    test_df[[sensi_col, 'Y', 'Y_pred', 'Y_pred_cc']].to_csv(cur_dir + '-'.join(['test_eval_single', str(seed), fair_setting]) + '.csv', index=False)

    setting_output = {}
    for pred_col, setting in zip(['Y_pred', 'Y_pred_cc'], ['A', 'C']):
        setting_output[setting] = eval_settings(test_df, sensi_col, pred_col)
    save_json(setting_output, cur_dir + '-'.join(['Single_eval', str(seed), fair_setting]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SingleCC, KAM-CAL, SingleCC+KAM-CAL over original test data")
    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(5)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(5)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
                                                                0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    on_same_thres = False
    res_path = '../intermediate/models/'

    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    if args.setting not in ['SingleCC', 'KAM-CAL', 'SingleCC+KAM-CAL']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')


    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, args.setting, on_same_thres, res_path])

        with Pool(cpu_count()) as pool:
            pool.starmap(eval_single_models, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')


