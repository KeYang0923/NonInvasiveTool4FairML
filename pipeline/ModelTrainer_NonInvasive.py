# Train and store XGBoost Tree models on the original data for SingleCC and KAM-CAL

import warnings
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from ModelTrainer_CAP import XgBoost
from utils import split
from joblib import dump

warnings.filterwarnings(action='ignore')

def XGB_trainer(data_name, y_col, sensi_col, seed, fair_setting,
                  res_path='../intermediate/models/',
                  verbose=True, data_path='../data/processed/'):
    cur_dir = res_path + data_name + '/'
    train_weight_df = pd.read_csv(cur_dir + '-'.join(['train_weights', str(seed), fair_setting]) + '.csv')
    weights = list(train_weight_df['weights'])

    output_file = fair_setting + '_xgb'

    if data_name in ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']:
        learner = XgBoost()
    else:
        raise ValueError('The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS]')


    input_df = pd.read_csv(data_path + data_name + '_dense.csv')

    train_df, validate_df, test_df = split(input_df, seed)

    features = [x for x in train_df.columns if x != y_col and x != sensi_col]

    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    val_data = validate_df[features]
    Y_val = np.array(validate_df[y_col])

    model = learner.fit(train_data, Y_train, features, sample_weight=weights)

    validate_df['Y'] = Y_val
    validate_df['Y_pred_scores'] = model.predict(val_data)

    dump(model, cur_dir + '-'.join(['model', output_file, str(seed)]) + '.joblib')
    validate_df[['Y', 'Y_pred_scores']].to_csv(cur_dir + '-'.join(['y_val', output_file, str(seed)]) + '.csv', index=False)

    train_df.to_csv(cur_dir + '-'.join(['train',  output_file, str(seed)]) + '.csv')  # keep index for sanity check of random splits
    validate_df.to_csv(cur_dir + '-'.join(['val',  output_file, str(seed)]) + '.csv', index=False)
    test_df.to_csv(cur_dir + '-'.join(['test',  output_file, str(seed)]) + '.csv', index=False)

    if verbose:
        score_train = learner.score(Y_train, model.predict(train_data))
        print('---' * 8, data_name, seed, '---' * 8)
        print(learner.scoring, "on train data: ", score_train)
        print('---' * 10, '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XGBoost Tree models on the original data for SingleCC and KAM-CAL")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=9,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(6)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(6)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
                                                                0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    if args.setting not in ['SingleCC', 'KAM-CAL', 'SingleCC+KAM-CAL']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

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
                tasks.append([data_name, y_col, sensi_col, seed, args.setting])
        with Pool(cpu_count()) as pool:
            pool.starmap(XGB_trainer, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')