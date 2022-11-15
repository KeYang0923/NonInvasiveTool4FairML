# Evaluate SingleCC and SingleCC+KAM-CAL over multiple degrees of intervention
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from utils import read_json, save_json

from ModelTrainer import generate_model_predictions
from utils import eval_settings
from ModelEvaluator_FairInterSingle import load_model_lr
warnings.filterwarnings('ignore')



def eval_SingleCC_degrees(data_name, seed, y_col, sensi_col, fair_setting, same_thres=False, res_path='../intermediate/models/', special_suffix=None):
    cur_dir = res_path + data_name + '/'
    if special_suffix is not None:
        fair_setting = fair_setting + '-' + special_suffix

    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed)]) + '.csv')
    features = [x for x in test_df.columns if x != y_col and x != sensi_col]
    orig_model = load_model_lr(cur_dir, seed)
    orig_thres = read_json(cur_dir+ '-'.join(['Thres', str(seed)]))['thres']

    cur_model = load_model_lr(cur_dir, seed, fair_setting)
    if same_thres:
        cur_thres = orig_thres
    else:
        cur_thres = read_json(cur_dir + '-'.join(['Thres', str(seed), fair_setting]))['thres']
    test_data = test_df[features]
    test_df['Y_pred_cc'] = generate_model_predictions(data_name, cur_model, test_data, cur_thres)
    test_df['Y_pred'] = generate_model_predictions(data_name, orig_model, test_data, orig_thres)
    test_df['Y'] = test_df[y_col]
    test_df[[sensi_col, 'Y', 'Y_pred', 'Y_pred_cc']].to_csv(cur_dir + '-'.join(['test_eval_fair', str(seed), fair_setting]) + '.csv', index=False)

    setting_output = {}
    for pred_col, setting in zip(['Y_pred', 'Y_pred_cc'], ['A', 'C']):
        setting_output[setting] = eval_settings(test_df, sensi_col, pred_col)
    save_json(setting_output, cur_dir + '-'.join(['Single_eval', str(seed), fair_setting]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SingleCC and SingleCC+KAM-CAL over multiple degrees of intervention")
    parser.add_argument("--data", type=str,
                        help="dataset to simulate all the intervention degree. Use 'all' for all the datasets. OR choose from [meps16, lawgpa, UFRGS] for different datasets.")

    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC, SingleCC+KAM-CAL] that represent SingleCC and SingleCC+KAM-CAL, respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running few number of executions and few intervention degrees
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    parser.add_argument("--exec_i", type=int, default=20,
                        help="number of intervention degrees. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lawgpa', 'UFRGS']

    if args.data is None:
        raise ValueError(
            'The input "data" is requried. Use "all" for all the datasets. OR choose from [meps16, lawgpa, UFRGS] for different datasets.')

    elif args.data == 'all':
        datasets = ['meps16', 'lawgpa']
        y_cols = ['Y' for i in range(2)]
        sensi_cols = ['C0' for i in range(2)]
    else:
        if args.data not in datasets:
            raise ValueError(
                'The input "data" is not supported. Choose from [meps16, lawgpa, UFRGS] for different datasets.')
        else:
            datasets = [args.data]
            y_cols = ['Y']
            sensi_cols = ['C0']

    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC, SingleCC+KAM-CAL] that represent SingleCC and SingleCC+KAM-CAL, respectively.')

    if args.setting not in ['SingleCC', 'SingleCC+KAM-CAL']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC, SingleCC+KAM-CAL] that represent SingleCC and SingleCC+KAM-CAL, respectively.')

    intervention_scales = [(0.1 + x / 10, 0.04 + 3 * x / 100) for x in range(20)]
    if args.exec_i is None:
        raise ValueError(
            'The input "exec_i" is requried. Use "--exec_i 1" for a single intervention degree.')
    elif type(args.exec_i) == str:
        raise ValueError(
            'The input "exec_i" requires integer. Use "--exec_i 1" for a single intervention degree.')
    else:
        i_exec = int(args.exec_i)
        intervention_scales = intervention_scales[:i_exec]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    on_same_thres = False
    res_path = '../intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for weight_i in intervention_scales:
                weight_suffix = '{}_{}'.format(weight_i[0], weight_i[1])
                for seed in seeds:
                    tasks.append([data_name, seed, y_col, sensi_col, args.setting, on_same_thres, res_path, weight_suffix])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(eval_SingleCC_degrees, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')

