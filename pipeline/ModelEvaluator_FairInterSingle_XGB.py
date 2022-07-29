# Compute evaluation metrics for comparison between SingleCC, KAM-CAL, and CAPUCHIN
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from joblib import load
from utils import read_json, save_json, eval_settings
from ModelTrainer import generate_model_predictions
warnings.filterwarnings('ignore')

def load_model_xgb(dir, seed, file_suffix):
    model = load(dir + '-'.join(['model', file_suffix, str(seed)]) + '.joblib')
    return model


def eval_XGB(data_name, seed, y_col, sensi_col, res_path='../intermediate/models/'):
    cur_dir = res_path + data_name + '/'

    setting_output = {}
    for setting, file_name in zip(['D', 'E', 'F'], ['CAP', 'KAM-CAL_xgb', 'SingleCC_xgb']):
        if file_name == 'CAP':
            test_df = pd.read_csv(cur_dir + '-'.join(['test', file_name, str(seed)]) + '.csv')
            setting_output[setting] = eval_settings(test_df, sensi_col, 'Y_pred')
        else:  # other two cases
            test_df = pd.read_csv(cur_dir + '-'.join(['test', file_name, str(seed)]) + '.csv')

            features = [x for x in test_df.columns if x != y_col and x != sensi_col]
            cur_model = load_model_xgb(cur_dir, seed, file_name)
            cur_thres = read_json(cur_dir + '-'.join(['Thres', file_name, str(seed)]))['thres']

            test_data = test_df[features]
            test_df['Y_pred'] = generate_model_predictions(data_name, cur_model, test_data, cur_thres)
            test_df['Y'] = test_df[y_col]

            test_df[[sensi_col, 'Y', 'Y_pred']].to_csv(
                cur_dir + '-'.join(['test_eval_fair', file_name, str(seed)]) + '.csv', index=False)

            setting_output[setting] = eval_settings(test_df, sensi_col, 'Y_pred')
    save_json(setting_output, cur_dir + '-'.join(['Single_eval', str(seed), 'XGBall']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for comparison between SingleCC, KAM-CAL, and CAPUCHIN")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()


    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(5)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(5)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
                                                                0xdeadcafe, 0xdeadbeef, 0xbeefcafe]


    res_path = '../intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, res_path])
        with Pool(cpu_count()) as pool:
            pool.starmap(eval_XGB, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')



