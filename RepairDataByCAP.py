# Process real datasets using CAPUCHINE
# REQUIRE the activation of virtual enviroment that installs AIF 360. See details at https://github.com/Trusted-AI/AIF360.
import warnings
import os, timeit, argparse

from CAPUCHIN.Core.indep_repair import Repair
from CAPUCHIN.Modules.InformationTheory.info_theo import Info
from TrainMLModels import read_json, save_json, make_folder
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

warnings.filterwarnings(action='ignore')


def generate_CAP_repaired_data(data_name, seed, res_path='../intermediate/models/', n_bins=5, repair_method='MF',
                                data_path='data/processed/', set_suffix ='S_1', sensi_col='A', y_col='Y'
                               ):
    repo_dir = res_path.replace('intermediate/models/', '')
    cur_dir = res_path + data_name + '/'

    train_df = pd.read_csv(cur_dir + '-'.join(['train', str(seed), set_suffix]) + '.csv')
    validate_df = pd.read_csv(cur_dir + '-'.join(['val', str(seed), set_suffix]) + '.csv')
    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed), set_suffix]) + '.csv')

    meta_info = read_json(repo_dir + '/' + data_path + data_name + '.json')
    n_cond_features = len(meta_info['continuous_features'])

    num_cols = ['X{}'.format(i) for i in range(1, n_cond_features + 1)]

    train_data = train_df[num_cols]
    val_data = validate_df[num_cols]
    test_data = test_df[num_cols]



    # bin the numerical attributes
    encode_flag = 'ordinal'
    bin_strategy = 'uniform'
    est = KBinsDiscretizer(n_bins=n_bins, encode=encode_flag, strategy=bin_strategy)
    est.fit(train_data)
    cat_train_data = est.transform(train_data)
    cat_val_data = est.transform(val_data)
    cat_test_data = est.transform(test_data)

    cat_train_df = pd.DataFrame(columns=num_cols, data=cat_train_data)
    cat_val_df = pd.DataFrame(columns=num_cols, data=cat_val_data)
    cat_test_df = pd.DataFrame(columns=num_cols, data=cat_test_data)

    cat_train_df[sensi_col] = train_df[sensi_col]
    cat_val_df[sensi_col] = validate_df[sensi_col]
    cat_test_df[sensi_col] = test_df[sensi_col]

    cat_train_df[y_col] = train_df[y_col]
    cat_val_df[y_col] = validate_df[y_col]
    cat_test_df[y_col] = test_df[y_col]

    # get original categorical features
    orig_train_df = pd.read_csv(cur_dir + '-'.join(['train', str(seed), set_suffix]) + '.csv')
    orig_val_df = pd.read_csv(cur_dir + '-'.join(['val', str(seed), set_suffix]) + '.csv')
    orig_test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed), set_suffix]) + '.csv')
    n_features = meta_info['n_features']

    cat_cols = ['X{}'.format(i) for i in range(n_cond_features + 1, n_features)]

    train_df = pd.concat([cat_train_df, orig_train_df[cat_cols]], axis=1)
    val_df = pd.concat([cat_val_df, orig_val_df[cat_cols]], axis=1)
    test_df = pd.concat([cat_test_df, orig_test_df[cat_cols]], axis=1)

    train_df.to_csv(cur_dir + '-'.join(['train', str(seed), 'bin']) + '.csv')  # keep index for sanity check of random splits
    val_df.to_csv(cur_dir + '-'.join(['val', str(seed), 'bin']) + '.csv', index=False)
    test_df.to_csv(cur_dir + '-'.join(['test', str(seed), 'bin']) + '.csv', index=False)

    save_json({'n_bins': n_bins, 'encode': encode_flag, 'strategy': bin_strategy}, '{}bin-{}.json'.format(cur_dir, seed))

    # run CAPUCHIN on train data
    start = timeit.default_timer()
    indep = [[sensi_col], [y_col], num_cols+cat_cols]
    inf = Info(train_df)
    X = indep[0]
    Y = indep[1]
    Z = indep[2]
    mi = inf.CMI(X, Y, Z)

    rep = Repair()
    if repair_method == 'sat':
        rep.from_file_indep_repair(cur_dir + 'train-', X, Y, Z, name_suffix='-bin', method=repair_method, n_parti=100,
                                   k=seed, sample_frac=1, insert_ratio='hard', conf_weight=1)
        rep.from_file_indep_repair(cur_dir + 'train-', X, Y, Z, name_suffix='-bin', method=repair_method, n_parti=100,
                                   k=seed, sample_frac=1, insert_ratio='soft', conf_weight=1)
    else:
        rep.from_file_indep_repair(cur_dir + 'train-', X, Y, Z, name_suffix='-bin', method=repair_method, n_parti=100,
                                   k=seed, sample_frac=1, insert_ratio=1, conf_weight=2000)

    end = timeit.default_timer()
    time = end - start
    save_json({'time': time}, '{}captime-{}.json'.format(cur_dir, seed))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Repair train data using CAPUCHIN")
    parser.add_argument("--bin_n", type=int, default=5,
                        help="number of binns in categorizing data. Required for CAPUCHIN. Default is 5 for all the datasets.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=10,
                        help="number of datasets over which the script is running. Default is 10 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['lsac', 'cardio', 'bank', 'meps16', 'credit', 'ACSE', 'ACSP', 'ACSH', 'ACSM', 'ACSI']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

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

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    for data_name in datasets:
        for seed in seeds:
            generate_CAP_repaired_data(data_name, seed, res_path, args.bin_n)
