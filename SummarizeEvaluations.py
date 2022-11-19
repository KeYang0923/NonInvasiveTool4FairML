import warnings
import os
from TrainMLModels import read_json, make_folder

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd

warnings.filterwarnings('ignore')


def extract_evaluations(data_name, seeds, models, res_path='../intermediate/models/',
               set_suffix='S_1', eval_path='eval/', ):

    repo_dir = res_path.replace('intermediate/models/', '')
    eval_path = repo_dir + '/' + eval_path
    make_folder(eval_path)

    group_eval_metrics = ['AUC', 'ACC', 'SR', 'BalAcc']
    overall_metrics = ['BalAcc', 'DI', 'EQDiff', 'AvgOddsDiff', 'SPDiff', 'FPRDiff', 'FNRDiff', 'ERRDiff']

    scc_weights = ['scc', 'scc', 'scc', 'omn', 'kam']
    scc_bases = ['one', 'kam', 'omn', 'one', 'one']

    res_df = pd.DataFrame(columns=['data', 'model', 'seed', 'method', 'group', 'metric', 'value'])
    cur_dir = res_path + data_name + '/'
    for model_i in models:
        if model_i == '':
            model_name = 'LR'
        if model_i == 'tr':
            model_name = 'TR'
            scc_weights = scc_weights + ['cap']
            scc_bases = scc_bases + ['one']
        for seed in seeds:
            # get multicc results
            eval_mcc_name = '{}{}eval-{}-{}-{}.json'.format(cur_dir, model_i, seed, set_suffix, 'mcc')
            if os.path.exists(eval_mcc_name):
                eval_res = read_json(eval_mcc_name)
                for mcc_i in ['MCC-MIN', 'MCC-W1', 'MCC-W2', 'SEP', 'ORIG']:
                    for group in ['all', 'G0', 'G1']:
                        base = [data_name, model_name, seed, mcc_i, group]
                        for metric_i in group_eval_metrics:
                            res_df.loc[res_df.shape[0]] = base + [metric_i, eval_res[mcc_i][group][metric_i]]
                    for metric_i in overall_metrics:
                        res_df.loc[res_df.shape[0]] = [data_name, model_name, seed, mcc_i, 'all'] + [metric_i, eval_res[mcc_i]['all'][metric_i]]

            # get single results
            for scc_i, base_i in zip(scc_weights, scc_bases):
                eval_scc_name = '{}{}eval-{}-{}-{}-{}.json'.format(cur_dir, model_i, seed, set_suffix, scc_i, base_i)
                if os.path.exists(eval_scc_name):
                    eval_res = read_json(eval_scc_name)
                    method_name = scc_i.upper() + base_i.upper()
                    for group in ['all', 'G0', 'G1']:
                        base = [data_name, model_name, seed, method_name, group]
                        for metric_i in group_eval_metrics:
                            res_df.loc[res_df.shape[0]] = base + [metric_i, eval_res[scc_i.upper()][group][metric_i]]
                    for metric_i in overall_metrics:
                        res_df.loc[res_df.shape[0]] = [data_name, model_name, seed, method_name, 'all'] + [metric_i, eval_res[scc_i.upper()]['all'][metric_i]]

    res_df.to_csv(eval_path+'res-{}.csv'.format(data_name), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract evaluation results")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="extract results for all the datasets as default. Otherwise, only extract the results for the input dataset.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['', 'tr'].")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()


    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]
    models = ['', 'tr']
    datasets = ['lsac', 'cardio', 'bank', 'meps16', 'credit', 'ACSE', 'ACSP', 'ACSH', 'ACSM', 'ACSI']
    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    else:
        raise ValueError('The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

    if args.model == 'all':
        pass
    elif args.model in models:
        models = [args.model]
    else:
        raise ValueError('The input "model" is not valid. CHOOSE FROM ["", "tr"].')

    if args.exec_n is None:
        raise ValueError('The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError('The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            tasks.append([data_name, seeds, models, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(extract_evaluations, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')