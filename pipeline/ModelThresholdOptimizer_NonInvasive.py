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
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=9,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()


    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'
    opt_obj = 'BalAcc'


    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC_xgb, KAM-CAL_xgb] that represent SingleCC and KAM-CAL, respectively.')

    if args.setting not in ['SingleCC_xgb', 'KAM-CAL_xgb']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC_xgb, KAM-CAL_xgb] that represent SingleCC and KAM-CAL, respectively.')

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
                tasks.append([data_name, seed, args.setting, res_path, opt_obj])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(compute_model_optimal_thresh_xgb, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')