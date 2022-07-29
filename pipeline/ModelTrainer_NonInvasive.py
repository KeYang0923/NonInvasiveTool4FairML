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
                  verbose=False, data_path='../data/processed/'):
    cur_dir = res_path + data_name + '/'
    train_weight_df = pd.read_csv(cur_dir + '-'.join(['train_weights', str(seed), fair_setting]) + '.csv')
    weights = list(train_weight_df['weights'])

    output_file = fair_setting + '_xgb'

    if data_name in ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']:
        learner = XgBoost()
    else:
        raise ValueError('The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit]')


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

    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(5)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(5)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
                                                                0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

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
                tasks.append([data_name, y_col, sensi_col, seed, args.setting])
        with Pool(cpu_count()) as pool:
            pool.starmap(XGB_trainer, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')