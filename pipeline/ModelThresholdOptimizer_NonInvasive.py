# Determine optimal threshold on validation set for XGBoost Tree models
import warnings
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from joblib import load
from utils import save_json
from ModelThresholdOptimizer import find_optimal_thres

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def load_xgb_model(dir, seed, file_suffix, return_val_data=True):
    model = load(dir + '-'.join(['model', file_suffix, str(seed)]) + '.joblib')
    Y_val_df = pd.read_csv(dir + '-'.join(['y_val', file_suffix, str(seed)]) + '.csv')
    if return_val_data:
        return model, Y_val_df
    else:
        return model

def compute_model_optimal_thresh_xgb(data_name, seed, file_suffix, path, opt_obj):
    cur_dir = path + data_name + '/'

    model, Y_val_df = load_xgb_model(cur_dir, seed, file_suffix)
    opt_thres = find_optimal_thres(Y_val_df, opt_obj=opt_obj)
    save_json(opt_thres, cur_dir + '-'.join(['Thres', file_suffix, str(seed)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Determine optimal threshold on validation set for XGBoost Tree models")

    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC_xgb, KAM-CAL_xgb] that represent SingleCC and KAM-CAL, respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()


    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
                                                                0xdeadcafe, 0xdeadbeef, 0xbeefcafe]
    res_path = '../intermediate/models/'
    opt_obj = 'BalAcc'


    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC_xgb, KAM-CAL_xgb] that represent SingleCC and KAM-CAL, respectively.')

    if args.setting not in ['SingleCC_xgb', 'KAM-CAL_xgb']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC_xgb, KAM-CAL_xgb] that represent SingleCC and KAM-CAL, respectively.')


    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds:
                tasks.append([data_name, seed, args.setting, res_path, opt_obj])
        with Pool(cpu_count()) as pool:
            pool.starmap(compute_model_optimal_thresh_xgb, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')