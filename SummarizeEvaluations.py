import warnings
import os
from PrepareData import read_json, make_folder

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd

warnings.filterwarnings('ignore')


def extract_evaluations(data_name, seeds, models, res_path='../intermediate/models/',
                        eval_path='eval/'):

    repo_dir = res_path.replace('intermediate/models/', '')
    eval_path = repo_dir + eval_path
    make_folder(eval_path)

    group_eval_metrics = ['AUC', 'ACC', 'SR', 'BalAcc']
    overall_metrics = ['BalAcc', 'DI', 'EQDiff', 'AvgOddsDiff', 'SPDiff', 'FPRDiff', 'FNRDiff', 'ERRDiff']

    scc_weights = ['scc', 'scc', 'omn', 'kam']
    scc_bases = ['one', 'kam', 'one', 'one']

    res_df = pd.DataFrame(columns=['data', 'model', 'seed', 'method', 'group', 'metric', 'value'])
    cur_dir = res_path + data_name + '/'
    for model_name in models:
        if model_name == 'tr':
            scc_weights = scc_weights + ['cap']
            scc_bases = scc_bases + ['one']
        for seed in seeds:
            # get multi results
            eval_mcc_name = '{}eval-{}-{}-{}.json'.format(cur_dir, model_name, seed, 'multi')
            if os.path.exists(eval_mcc_name):
                eval_res = read_json(eval_mcc_name)
                for mcc_i in ['MCC-MIN', 'MCC-W1', 'MCC-W2', 'SEP', 'ORIG']:
                    for group in ['all', 'G0', 'G1']:
                        base = [data_name, model_name.upper(), seed, mcc_i, group]
                        for metric_i in group_eval_metrics:
                            res_df.loc[res_df.shape[0]] = base + [metric_i, eval_res[mcc_i][group][metric_i]]
                    for metric_i in overall_metrics:
                        res_df.loc[res_df.shape[0]] = [data_name, model_name.upper(), seed, mcc_i, 'all'] + [metric_i, eval_res[mcc_i]['all'][metric_i]]
            else:
                print('--> no eval for', eval_mcc_name)
            # get single results
            for weight_i, base_i in zip(scc_weights, scc_bases):
                eval_single_name = '{}eval-{}-{}-{}-{}.json'.format(cur_dir, model_name, seed, weight_i, base_i)
                if os.path.exists(eval_single_name):
                    eval_res = read_json(eval_single_name)
                    method_name = weight_i.upper() + '-' + base_i.upper()
                    for group in ['all', 'G0', 'G1']:
                        base = [data_name, model_name.upper(), seed, method_name, group]
                        for metric_i in group_eval_metrics:
                            res_df.loc[res_df.shape[0]] = base + [metric_i, eval_res[weight_i.upper()][group][metric_i]]
                    for metric_i in overall_metrics:
                        res_df.loc[res_df.shape[0]] = [data_name, model_name.upper(), seed, method_name, 'all'] + [metric_i, eval_res[weight_i.upper()]['all'][metric_i]]
                else:
                    print('--> no eval for', eval_single_name)
    res_df.to_csv(eval_path+'res-{}.csv'.format(data_name), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract evaluation results")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")

    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac', 'bank', 'cardio', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']

    seeds = [88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]

    models = ['lr', 'tr']

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
    if args.model == 'all':
        pass
    elif args.model in models:
        models = [args.model]
    else:
        raise ValueError('The input "model" is not valid. CHOOSE FROM ["lr", "tr"].')

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