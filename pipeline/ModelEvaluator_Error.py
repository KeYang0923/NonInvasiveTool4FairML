# Evaluate models over erroneous test data
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd

from utils import read_json, save_json
from ModelTrainer import generate_model_predictions
from utils import eval_settings
from ModelEvaluator_AllDegrees import load_model_lr
warnings.filterwarnings('ignore')



def reverse_sensi_col(x, cand_list):
    if x.iloc[0] in cand_list:
        return abs(x.iloc[1] - 1)
    else:
        return x.iloc[1]

def eval_erroneous_data(data_name, seed, y_col, sensi_col, eval_setting, fair_setting, res_path='../intermediate/models/'):
    cur_dir = res_path + data_name + '/'

    if 'group' in eval_setting: # multiple models over error data
        test_drift_df = pd.read_csv(cur_dir + '-'.join(['test_eval_multi', str(seed), fair_setting]) + '.csv')
        # read the indexes for the wrong group membership
        # use the original group membership rather than the wrong one in evaluation
        error_indexes = read_json('{}test_errorID-{}-{}.csv'.format(cur_dir, seed, fair_setting))['Error_ID']
        test_drift_df['ID_temp'] = test_drift_df.index
        test_drift_df[sensi_col] = test_drift_df[['ID_temp', sensi_col]].apply(lambda x: reverse_sensi_col(x, error_indexes), axis=1)

        setting_output_drift = {}
        for pred_col, setting in zip(['Y_pred', 'Y_pred_G0', 'Y_pred_G1', 'Y_pred_en', 'Y_pred_en_cc'],
                                     ['A', 'A0', 'A1', 'E', 'F']):
            setting_output_drift[setting] = eval_settings(test_drift_df, sensi_col, pred_col)

        save_json(setting_output_drift, cur_dir + '-'.join(['Multi_eval', str(seed), fair_setting]))

    else: # single model over error data
        # print('Activate singleCC over erroneous data!')
        test_df = pd.read_csv('{}test-{}-{}.csv'.format(cur_dir, seed, fair_setting))

        setting_output = {}
        features = [x for x in test_df.columns if x != y_col and x != sensi_col]
        cur_model = load_model_lr(cur_dir, seed, 'cc')
        cur_thres = read_json(cur_dir + '-'.join(['Thres', str(seed), 'cc']))['thres']

        test_data = test_df[features]
        test_df['Y_pred'] = generate_model_predictions(data_name, cur_model, test_data, cur_thres)
        test_df['Y'] = test_df[y_col]

        test_df[[sensi_col, 'Y', 'Y_pred']].to_csv(cur_dir + '-'.join(['test_eval_single', str(seed), fair_setting]) + '.csv', index=False)

        # read the indexes for the wrong group membership
        # use the original group membership rather than the wrong one in evaluation
        error_indexes = read_json('{}test_errorID-{}-{}.csv'.format(cur_dir, seed, fair_setting))['Error_ID']
        test_df['ID_temp'] = test_df.index
        test_df[sensi_col] = test_df[['ID_temp', sensi_col]].apply(lambda x: reverse_sensi_col(x, error_indexes), axis=1)

        setting_output['C'] = eval_settings(test_df, sensi_col, 'Y_pred')
        save_json(setting_output, cur_dir + '-'.join(['Single_eval', str(seed), fair_setting]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate models over erroneous test data")
    parser.add_argument("--data", type=str,
                        help="dataset to simulate all the error rates. Use 'all' for all the datasets. OR choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS] for different datasets.")
    parser.add_argument("--model", type=str,
                        help="setting of evaluation for single or group models, if 'group' run for all the models in MultiCC. Otherwise, only for the single model in SingleCC.")

    parser.add_argument("--setting", type=str,
                        help="setting of evaluation for single or group models, if 'all' run for all the error rates. Otherwise, only for the input setting, e.g., error0.15.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running few number of executions and few error rates
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    parser.add_argument("--exec_k", type=int, default=30,
                        help="number of error rates. Default is 30.")
    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    y_cols_mapping = {'adult': 'Income Binary', 'german': 'credit', 'compas': 'two_year_recid',
                      'cardio': 'Y', 'bank': 'Y', 'meps16': 'Y', 'lawgpa': 'Y', 'credit': 'Y', 'UFRGS': 'Y'}

    sensi_cols_mapping = {'adult': 'sex', 'german': 'age', 'compas': 'race',
                          'cardio': 'C0', 'bank': 'C0', 'meps16': 'C0', 'lawgpa': 'C0', 'credit': 'C0', 'UFRGS': 'C0'}

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'

    if args.data is None:
        raise ValueError('The input "data" is requried. Use "all" for all the datasets. OR choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit] for different datasets.')

    elif args.data == 'all':
        datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
        y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(6)]
        sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(6)]

    else:
        if args.data not in datasets:
            raise ValueError(
                'The input "data" is not supported. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS] for different datasets.')
        else:
            datasets = [args.data]
            y_cols = [y_cols_mapping[args.data]]
            sensi_cols = [sensi_cols_mapping[args.data]]

    if args.model is None:
        raise ValueError(
            'The input "model" is not required. Choose from [single, group].')
    elif args.model not in ['single', 'group']:
        raise ValueError(
            'The input "model" is not supported. Choose from [single, group].')


    if args.setting is None:
        raise ValueError('The input "setting" is requried. Use "all" for multiple error rates. OR "error"+0.1 for the example of 10% errors simulated in the test set.')

    elif 'error' not in args.setting:
        errors_k = [x / 100 for x in range(1, 30)]
        if args.setting != 'all':
            raise ValueError(
                'The input "setting" is not supported.  Use "all" for multiple error rates. OR "error"+0.1 for the example of 10% errors simulated in the test set.')
        else:
            if args.exec_k is None:
                raise ValueError(
                    'The input "exec_k" is requried. Use "--exec_k 1" for a single error rate.')
            elif type(args.exec_k) == str:
                raise ValueError(
                    'The input "exec_k" requires integer. Use "--exec_k 1" for a single error rate.')
            else:
                k_exec = int(args.exec_k)
                errors_k = errors_k[:k_exec]
    else: # erroneous case under single error rate
        if args.setting not in ['error0.2', 'error0.15', ' error0.1', 'error0.05', 'error0.01']:
            raise ValueError(
                'The input "setting" is not supported. Use "all" for multiple error rates. OR choose from [error0.1, error0.15, error0.2, error0.01, error0.05]. OR run "ErrorDataSimulator.py" for new desired rates.')
        else:
            errors_k = [float(args.setting.replace('error', ''))]


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
        for error_k in errors_k:
            cc_setting = 'error{:.2f}'.format(error_k)
            for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
                for seed in seeds:
                    tasks.append([data_name, seed, y_col, sensi_col, args.model, cc_setting, res_path])
        with Pool(cpu_count()) as pool:
            pool.starmap(eval_erroneous_data, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')
