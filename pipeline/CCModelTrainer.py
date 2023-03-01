# Optimize the CC's input and learn rules from training data using CCs

import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import prose.datainsights as di
from utils import format_print, save_json, split

warnings.filterwarnings('ignore')

def estimate_density(input_df, y_col, sensi_col, dense_cols, n_groups=2, n_labels=2,
                     dense_h=0.1, dense_kernal='tophat', algorithm='auto'):
    res_df = input_df.copy()

    for group_i in range(n_groups):
        for c_i in range(n_labels):
            dense_col_name = '_'.join(['density', 'G' + str(group_i), 'L' + str(c_i)])
            dense_train = res_df[(res_df[sensi_col] == group_i) & (res_df[y_col] == c_i)].copy()
            dense_X = dense_train[dense_cols].to_numpy()
            kde = KernelDensity(bandwidth=dense_h, kernel=dense_kernal, algorithm=algorithm)
            kde.fit(dense_X)
            res_df[dense_col_name] = kde.score_samples(res_df[dense_cols].to_numpy())
    return res_df


def combine_violation(x):
    idx_map = {'00': 0, '01': 1, '10': 2, '11': 3}
    cur = '{}{}'.format(int(x.iloc[0]), int(x.iloc[1]))
    return x.iloc[2 + idx_map[cur]]

def return_identify_zero_violation(dense_df, test_df, sensi_col, y_col, group_names, label_names, cc_cols,
                                dense_n=2000, n_groups=2, n_labels=2, verbose=True,
                                save_print_f=None):
    vio_df = dense_df.copy()
    test_vio = test_df.copy()
    if verbose:
        if save_print_f is not None:
            f = open(save_print_f + '.txt', 'w')
        else:
            f = None
    core_conform_index = {}
    opp_conform_index = {}

    cc_models = {}
    for group_i, group_name in zip(range(n_groups), group_names):
        for label_i, label_name in zip(range(n_labels), label_names):
            dense_col_name = '_'.join(['density', 'G' + str(group_i), 'L' + str(label_i)])

            di_train = vio_df[(vio_df[sensi_col] == group_i) & (vio_df[y_col] == label_i)]

            di_train.sort_values(by=[dense_col_name], ascending=False, inplace=True)

            di_train_dense = di_train.iloc[:dense_n, :] # dense_n points with highest density score
            assertions_i = di.learn_assertions(di_train_dense[cc_cols], max_self_violation=1.0)


            result_i = assertions_i.evaluate(vio_df[cc_cols], explanation=True, normalizeViolation=True)
            vio_df['_'.join(['vio_G' + str(group_i), 'L' + str(label_i)])] = result_i.row_wise_violation_summary['violation']

            test_result_i = assertions_i.evaluate(test_vio[cc_cols], explanation=True, normalizeViolation=True)
            test_vio['vio_by_' + dense_col_name] = test_result_i.row_wise_violation_summary['violation']


            result_i = assertions_i.evaluate(di_train[cc_cols], explanation=True, normalizeViolation=True)
            di_train['vio_by_' + dense_col_name] = result_i.row_wise_violation_summary['violation']

            cur_core = di_train[di_train['vio_by_' + dense_col_name] == 0].copy()
            core_conform_index[str(group_i) + '_' + str(label_i)] = list(cur_core.index)

            cc_models[dense_col_name] = assertions_i

            format_print('For %s = %s & %s = %s'% (sensi_col, group_name, y_col, label_name), output_f=f)
            result_train = assertions_i.evaluate(di_train_dense[cc_cols], explanation=True, normalizeViolation=True)
            format_print(assertions_i, output_f=f)
            format_print('\n', output_f=f)

            format_print('Violation on training %4.3f'%result_train.avg_violation, output_f=f)
            format_print('Zero-violated (Core) samples in training %d out of %d \n' % (cur_core.shape[0], di_train.shape[0]), output_f=f)
            format_print('\n', output_f=f)

            label_test_df = vio_df[(vio_df[sensi_col] == group_i) & (vio_df[y_col] != label_i)].copy()
            result_label = assertions_i.evaluate(label_test_df[cc_cols], explanation=True, normalizeViolation=True)
            format_print('Violation on opposite label %4.3f'%result_label.avg_violation, output_f=f)
            label_test_df['vio_by_' + dense_col_name] = result_label.row_wise_violation_summary['violation']
            format_print('Zero-violated samples in opposite label %d out of %d' % (label_test_df[label_test_df['vio_by_' + dense_col_name] == 0].shape[0], label_test_df.shape[0]), output_f=f)
            format_print('\n', output_f=f)
            opp_conform_train = label_test_df[label_test_df['vio_by_' + dense_col_name] == 0].copy()
            opp_conform_index[str(group_i) + '_' + str(label_i)] = list(opp_conform_train.index)

            group_test_df = vio_df[(vio_df[sensi_col] != group_i) & (vio_df[y_col] == label_i)].copy()
            result_group = assertions_i.evaluate(group_test_df[cc_cols], explanation=True, normalizeViolation=True)
            group_test_df['vio_by_' + dense_col_name] = result_group.row_wise_violation_summary['violation']

            format_print('Violation on other group %4.3f'%result_group.avg_violation, output_f=f)
            zero_vio_group_test = group_test_df[group_test_df['vio_by_' + dense_col_name] == 0]
            format_print('Zero-violated samples in other group %d out of %d' % (zero_vio_group_test.shape[0], group_test_df.shape[0]), output_f=f)
            format_print('\n', output_f=f)

            both_test = vio_df[(vio_df[sensi_col] != group_i) & (vio_df[y_col] != label_i)].copy()
            result_both = assertions_i.evaluate(both_test[cc_cols], explanation=True, normalizeViolation=True)
            both_test['vio_by_' + dense_col_name] = result_both.row_wise_violation_summary['violation']

            format_print('Violation on other group opposite label %4.3f'%result_both.avg_violation, output_f=f)
            format_print('Zero-violated samples in other group opposite label %d out of %d' % (both_test[both_test['vio_by_' + dense_col_name] == 0].shape[0], both_test.shape[0]), output_f=f)
            format_print('\n', output_f=f)

    # if save_print_f is not None:
    #     f.close()
    # return cc_models, core_conform_index, opp_conform_index
    vio_df['vio_cc'] = vio_df[[sensi_col, y_col, 'vio_G0_L0', 'vio_G0_L1', 'vio_G1_L0', 'vio_G1_L1']].apply(lambda x: combine_violation(x), axis=1)
    return vio_df, test_vio


def learn_cc_models(data_name, seed, y_col, sensi_col, name_mapping, cc_skiped_cols, dense_skiped_cols,
                    res_path='../intermediate/models/',
                    n_groups=2, n_labels=2,
                    data_path='../data/processed/'):
    if data_name in ['german', 'lawgpa']:
        dense_n = 500
    elif data_name in ['adult', 'compas', 'cardio', 'bank', 'meps16', 'credit', 'UFRGS']:
        dense_n = 2000
    else:
        raise ValueError('The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS]')

    cur_dir = res_path + data_name + '/'
    # cc_print_file = cur_dir + '-'.join(['train_CC', str(seed)])

    group_names, label_names, dense_kernal = name_mapping

    input_df = pd.read_csv(data_path + data_name + '_dense.csv')
    # input_df = pd.read_csv(data_path + data_name + '_dense_opp.csv')

    train_df, validate_df, test_df = split(input_df, seed)
    dense_cols = sorted(list(set(train_df.columns) - set(dense_skiped_cols+[sensi_col, y_col])))

    # estimate desnsity
    dense_train = estimate_density(train_df, sensi_col, y_col, dense_cols, dense_kernal=dense_kernal)

    cc_cols = sorted(list(set(train_df.columns) - set(cc_skiped_cols+[sensi_col, y_col])))
    vio_train, test_vio = return_identify_zero_violation(dense_train, test_df, sensi_col, y_col, group_names, label_names, cc_cols, dense_n=dense_n)


    # cc_models, core_conform_index, opp_conform_index = return_identify_zero_violation(dense_train, sensi_col, y_col, group_names, label_names,
    #                                            cc_cols, dense_n=dense_n, save_print_f=cc_print_file)
    # total_n = train_df.shape[0]
    #
    # # save the violation on test data based on sensi_col + y_col
    # weights = {}
    # for group_i, group_name in zip(range(n_groups), group_names):
    #     n_g = train_df[train_df[sensi_col] == group_i].shape[0]
    #
    #     for label_i, label_name in zip(range(n_labels), label_names):
    #         n_label = train_df[train_df[y_col] == label_i].shape[0]
    #
    #         dense_col_name = '_'.join(['density', 'G' + str(group_i), 'L' + str(label_i)])
    #         cc_train = train_df[(train_df[sensi_col] == group_i) & (train_df[y_col] == label_i)] #.copy()
    #
    #         cur_assertion = cc_models[dense_col_name]
    #
    #         prop_weight_base = n_label * n_g / (total_n * cc_train.shape[0])
    #
    #         result_i = cur_assertion.evaluate(test_df[cc_cols], explanation=True, normalizeViolation=True)
    #         test_df['vio_by_' + dense_col_name] = result_i.row_wise_violation_summary['violation']
    #
    #         cur_core_conform = core_conform_index[str(group_i) + '_' + str(label_i)]
    #         cur_opp_conform = opp_conform_index[str(group_i) + '_' + str(label_i)]
    #
    #         weights[str(group_i) + '_' + str(label_i)] = {'prop_weight': prop_weight_base, 'core_conform': cur_core_conform, 'opp_conform': cur_opp_conform}

    # test_df.to_csv(cur_dir+'-'.join(['test_violation', str(seed)])+'.csv', index=False)

    vio_train.to_csv(cur_dir + '-'.join(['train_violation', str(seed)]) + '.csv', index=False)
    test_vio.to_csv(cur_dir+'-'.join(['test_violation', str(seed)])+'.csv', index=False)

    # dense_train.to_csv(cur_dir + '-'.join(['train_density', str(seed)]) + '.csv', index=False)
    # # save sample index of train from the four cases
    # save_json(weights, cur_dir+'-'.join(['train_weights', str(seed)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimize CC and learn rules from training data using CCs")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=9,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    name_cols = {'adult': (['Female', 'Male'], ['<=50K', '>50K'], 'tophat'),
                 'german': (['Young', 'Old'], ['Bad Credit', 'Good Credit'], 'tophat'),
                 'compas': (['Not Caucasian', 'Caucasian'], ['Did recid.', 'No recid.'], 'tophat'),
                 'cardio': (['Young', 'Old'], ['Not diagnosed', 'Diagnosed'], 'gaussian'),
                 'bank': (['Married&Divorced', 'Single'], ['Declined', 'Approval'], 'tophat'),
                 'meps16': (['Not White', 'White'], ['Low utilization', 'High utilization'], 'tophat'),
                 'lawgpa': (['Black', 'White'], ['low', 'high'], 'gaussian'),
                 'credit': (['Old', 'Young'], ['no', 'yes'], 'tophat'),
                 'UFRGS': (['female', 'male'], ['lowGPA', 'highGPA'], 'gaussian')}

    # skip the categorical attributes and useless numerical ones in the input CC
    cc_skip_cols_mapping = {'adult': ['race'],
                    'german': ['sex'],
                    'compas': ['sex'],
                    'cardio': ['X' + str(i) for i in range(4, 18)],
                    'bank': ['X' + str(i) for i in range(5, 16)],
                    'meps16': ['X' + str(i) for i in range(5, 138)],
                    'lawgpa': [],
                    'credit': [],
                    'UFRGS': []
                    }
    dense_skip_cols_mapping = {'adult': ['race'],
                    'german': ['sex'],
                    'compas': ['sex'],
                    'cardio': ['X' + str(i) for i in range(4, 18)],
                    'bank': ['X' + str(i) for i in range(5, 16)],
                    'meps16': ['X' + str(i) for i in range(5, 138)],
                    'lawgpa': [],
                    'credit': [],
                    'UFRGS': []
                    }


    # datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    # y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(6)]
    # sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(6)]

    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    datasets = ['german']
    y_cols = ['credit']
    sensi_cols = ['age']
    seeds = [1]

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
            name_mapping = name_cols[data_name]
            cc_skip_cols = cc_skip_cols_mapping[data_name]
            dense_skip_cols = dense_skip_cols_mapping[data_name]
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, name_mapping, cc_skip_cols, dense_skip_cols, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(learn_cc_models, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')