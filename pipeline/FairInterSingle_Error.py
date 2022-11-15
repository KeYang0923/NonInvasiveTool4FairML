# Use fairness interventions on training data with a single model and retrain the same ML models under a fixed interention degree
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from joblib import dump
from ModelTrainer import AIFLogisticRegression, SKLogisticRegression, generate_model_predictions
from utils import read_json

warnings.filterwarnings('ignore')

def set_prop_weights(x, weights_mapping):
    return weights_mapping[str(int(x.iloc[0])) + '_'+ str(int(x.iloc[1]))]

def adjust_weight_unit(target_down_unit, start_up_unit, start_up_n, used_down_n, step=0.01):
    while target_down_unit > 1.0:
        start_up_unit = start_up_unit - step
        target_down_unit = start_up_unit * start_up_n / used_down_n
    return target_down_unit, start_up_unit

def retrain_models_by_faircc(data_name, seed, y_col, sensi_col, fair_setting, weights_input=None, error_suffix=None,
                             output_path='../intermediate/models/', special_suffix=None,
                             n_groups=2, n_labels=2, train_index_col='Unnamed: 0'):

    out_dir = output_path + data_name + '/'
    # load the weights from CCs
    cc_weights = read_json(out_dir + '-'.join(['train_weights', str(seed), error_suffix]))

    # retrain the model using the weights set up based on the learned samples
    train_df = pd.read_csv(out_dir + '-'.join(['train', str(seed), error_suffix]) + '.csv', index_col=train_index_col)

    validate_df = pd.read_csv(out_dir + '-'.join(['val', str(seed)]) + '.csv')
    test_df = pd.read_csv(out_dir + '-'.join(['test', str(seed)]) + '.csv')

    # features = [x for x in test_df.columns if x != y_col and x != sensi_col]
    features = [x for x in test_df.columns if x != y_col]

    np.random.seed(seed)
    if data_name in ['adult', 'german', 'compas']:
        learner = AIFLogisticRegression()
    elif data_name in ['cardio', 'bank', 'lawgpa', 'meps16', 'credit', 'UFRGS']:
        learner = SKLogisticRegression()
    else:
        raise ValueError('The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS]')

    # initiate weight base
    if 'KAM-CAL' in fair_setting: # set the weights using KAM-CAL
        prop_weights_mapping = {}
        for group_i in range(n_groups):
            for label_i in range(n_labels):
                prop_weights_mapping[str(group_i) + '_' + str(label_i)] = cc_weights[str(group_i) + '_' + str(label_i)]['prop_weight']
        train_df['weights'] = train_df[[sensi_col, y_col]].apply(lambda x: set_prop_weights(x, prop_weights_mapping), axis=1)
    else:
        train_df['weights'] = 1.0

    if 'SingleCC' in fair_setting:
        # read core points identified by CC
        cond_n = []
        for group_i in range(n_groups):
            for label_i in range(n_labels):
                gl_df = train_df[(train_df[sensi_col]==group_i) & (train_df[y_col]==label_i)]
                cond_n.append(gl_df.shape[0])
                key_str = str(group_i) + str(label_i)
                core_samples_index = cc_weights[str(group_i) + '_' + str(label_i)]['core_conform']
                train_df[key_str + '_core'] = False
                train_df.loc[core_samples_index, key_str + '_core'] = True

        n_mi_neg, n_mi_pos, n_ma_neg, n_ma_pos = cond_n[0], cond_n[1], cond_n[2], cond_n[3]
        n_mi_up, n_mi_down, n_ma_up, n_ma_down = train_df['01_core'].sum(), (n_mi_neg - train_df['00_core'].sum()), train_df['10_core'].sum(),  (n_ma_pos - train_df['11_core'].sum())

        if n_mi_down < 0.3*n_mi_neg:  # if not enough points contribute to samples that need to downgrade the weights
            n_mi_down = int(0.3*n_mi_neg)
        if n_ma_down < 0.3*n_ma_pos:  # if not enough points contribute to down samples that need to downgrade the weights
            n_ma_down = int(0.3*n_ma_pos)

        if weights_input is not None:
            w_mi_up_unit, w_ma_up_unit = weights_input[data_name]
        else: # default intervention degree
            w_mi_up_unit, w_ma_up_unit = 1.5, 0.5

        w_mi_down_unit, w_ma_down_unit = w_mi_up_unit * n_mi_up / n_mi_down, w_ma_up_unit * n_ma_up / n_ma_down

        if w_mi_down_unit > 1.0:
            n_mi_down = n_mi_neg
            w_mi_down_unit = w_mi_up_unit * n_mi_up / n_mi_down
            w_mi_down_unit, w_mi_up_unit = adjust_weight_unit(w_mi_down_unit, w_mi_up_unit, n_mi_up, n_mi_down)
        if w_ma_down_unit > 1.0:
            n_ma_down = n_ma_pos
            w_ma_down_unit = w_ma_up_unit * n_ma_up / n_ma_down
            w_ma_down_unit, w_ma_up_unit = adjust_weight_unit(w_ma_down_unit, w_ma_up_unit, n_ma_up, n_ma_down)

        w_mi_up_index = train_df[train_df['01_core']].index
        w_ma_up_index = train_df[train_df['10_core']].index

        w_mi_down_index = train_df[~(train_df['00_core'] | train_df['01_core'] | train_df['10_core'])].index
        if len(w_mi_down_index) < n_mi_down:
            n_mi_down = len(w_mi_down_index)
            w_mi_down_unit = w_mi_up_unit * n_mi_up / n_mi_down
            w_mi_down_unit, w_mi_up_unit = adjust_weight_unit(w_mi_down_unit, w_mi_up_unit, n_mi_up, n_mi_down)
        else:
            w_mi_down_index = list(w_mi_down_index)[:n_mi_down]

        assert len(w_mi_down_index) == n_mi_down

        w_ma_down_index = train_df[~(train_df['11_core'] | train_df['10_core'])].index
        w_ma_down_index = set(w_ma_down_index) - set(w_mi_down_index) - set(w_mi_up_index)
        if len(w_ma_down_index) < n_ma_down: # update n_ma_down as the data is not able to support it
            n_ma_down = len(w_ma_down_index)
            w_ma_down_unit = w_ma_up_unit * n_ma_up / n_ma_down
            w_ma_down_unit, w_ma_up_unit = adjust_weight_unit(w_ma_down_unit, w_ma_up_unit, n_ma_up, n_ma_down)
        else:
            w_ma_down_index = list(w_ma_down_index)[:n_ma_down]
        assert len(w_ma_down_index) == n_ma_down

        train_df.loc[w_mi_up_index, 'weights'] += w_mi_up_unit
        train_df.loc[w_ma_up_index, 'weights'] += w_ma_up_unit
        train_df.loc[w_mi_down_index, 'weights'] -= w_mi_down_unit
        train_df.loc[w_ma_down_index, 'weights'] -= w_ma_down_unit

    train_weights = list(train_df['weights'])


    assert len(train_weights) == train_df.shape[0]
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    val_data = validate_df[features]
    Y_val = np.array(validate_df[y_col])

    model = learner.fit(train_data, Y_train, features, sample_weight=train_weights)

    validate_df['Y'] = Y_val
    validate_df['Y_pred_scores'] = generate_model_predictions(data_name, model, val_data)

    dump(model, out_dir + '-'.join(['model', str(seed), fair_setting, error_suffix]) + '.joblib')
    validate_df[['Y', 'Y_pred_scores']].to_csv(out_dir + '-'.join(['y_val', str(seed), fair_setting, error_suffix]) + '.csv', index=False)
    train_df.to_csv(out_dir + '-'.join(['train_weights', str(seed), fair_setting, error_suffix]) + '.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply fairness intervention on training data with a single ML model and retrain the model under a fixed interention degree")
    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.")
    parser.add_argument("--error", type=str,
                        help="setting of error simulation. Choose from [error0.05, error0.10, error0.15, error0.20, error0.25, error0.30] that represent different error rates in the training data respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=1,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    # optimized intervention degree in SingleCC
    weights_adjust = {'adult': [1.8, 0.4],
                      'german': [0.35, 0.1],
                      'compas': [0.15, 0.05],
                      'cardio': [2.2, 1.0],
                      'bank': [0.3, 0.1],
                      'meps16': [1.5, 0.5],
                      'lawgpa': [1.5, 0.5],
                      'credit': [1.95, 0.95],
                      'UFRGS': [0.3, 0.1]
                      }
    # UFRGS [0.3, 0.1]

    # datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    # y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(6)]
    # sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(6)]

    datasets = ['lawgpa']
    y_cols = ['Y']
    sensi_cols = ['C0']

    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]
    seeds = [6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

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


    res_path = '../intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, args.setting, weights_adjust, args.error, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(retrain_models_by_faircc, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')