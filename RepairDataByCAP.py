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


def generate_CAP_repaired_data(data_name, seed, res_path='../intermediate/models/',repair_method='MF',
                                data_path='data/processed/', sensi_col='A', y_col='Y'
                               ):
    repo_dir = res_path.replace('intermediate/models/', '')
    cur_dir = res_path + data_name + '/'
    meta_info = read_json(repo_dir + '/' + data_path + data_name + '.json')
    n_cond_features = len(meta_info['continuous_features'])
    num_cols = ['X{}'.format(i) for i in range(1, n_cond_features + 1)]
    n_features = meta_info['n_features']
    cat_cols = ['X{}'.format(i) for i in range(n_cond_features + 1, n_features)]

    train_df = pd.read_csv('{}train-{}-bin.csv'.format(cur_dir, seed))

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
    save_json({'time': time}, '{}time-cap-{}.json'.format(cur_dir, seed))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Repair train data using CAPUCHIN")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()


    datasets = ['meps16', 'lsac', 'bank', 'cardio', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]

    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    else:
        raise ValueError(
            'The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

    if args.set_n is not None:
        if type(args.set_n) == str:
            raise ValueError(
                'The input "set_n" requires integer. Use "--set_n 1" for running over a single dataset.')
        else:
            n_datasets = int(args.set_n)
            if n_datasets < 0:
                datasets = datasets[n_datasets:]
            elif n_datasets > 0:
                datasets = datasets[:n_datasets]
            else:
                raise ValueError(
                    'The input "set_n" requires non-zero integer. Use "--set_n 1" for running over a single dataset.')

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
            generate_CAP_repaired_data(data_name, seed, res_path)
