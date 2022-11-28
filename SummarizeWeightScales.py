import warnings
from PrepareData import read_json

import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
import os

warnings.filterwarnings(action='ignore')

def extract_impact_weight_scales(data_name, seeds, models, res_path='../intermediate/models/',
                                 eval_path='eval/'):

    repo_dir = res_path.replace('intermediate/models/', '')
    eval_path = repo_dir + eval_path

    cur_dir = res_path + data_name + '/'
    scc_weights = ['scc', 'scc', 'omn']
    scc_bases = ['one', 'kam', 'one']

    res_df = pd.DataFrame(columns=['data', 'model', 'seed', 'method', 'degree', 'BalAcc', 'SPDiff'])

    for model_i in models:
        if model_i == 'lr':
            extra_weights = ['kam']
        if model_i == 'tr':
            extra_weights = ['kam', 'cap']
        model_name = model_i.upper()
        for seed in seeds:
            for reweigh_method, weight_base in zip(scc_weights, scc_bases):
                degree_file ='{}degrees-{}-{}-{}-{}.txt'.format(cur_dir, model_name, seed, reweigh_method, weight_base)
                if os.path.exists(degree_file):
                    f = open(degree_file, "r")
                    while (True):
                        line = f.readline()
                        if not line:
                            break
                        cur_res = line.strip().replace('---', '').split(' ')
                        res_df.loc[res_df.shape[0]] = [data_name, model_name, seed, reweigh_method.upper() + '-' + weight_base.upper(), float(cur_res[0]), float(cur_res[1]), float(cur_res[2])]
                else:
                    print('---> no ', degree_file)
            # get the results from the methods without degrees
            for reweigh_method in extra_weights:
                weight_base = 'one'
                eval_file = '{}eval-{}-{}-{}-{}.json'.format(cur_dir, model_i, seed, reweigh_method, weight_base)
                if os.path.exists(eval_file):
                    eval_res = read_json(eval_file)

                    acc = eval_res[reweigh_method.upper()]['all']['BalAcc']
                    sp = eval_res[reweigh_method.upper()]['all']['SPDiff']
                    for degree_i in [x / 100 for x in range(1, 201)]:
                        res_df.loc[res_df.shape[0]] = [data_name, model_name, seed, reweigh_method.upper() + '-' + weight_base.upper(), degree_i, acc, sp]
                else:
                    print('---> no ', eval_file)
    res_df.to_csv(eval_path+'degree-{}.csv'.format(data_name), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract evaluation results")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="extract results for all the datasets as default. Otherwise, only extract the results for the input dataset.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['lsac', 'meps16', 'ACSP', 'ACSH']

    seeds = [1, 12345, 6, 2211, 15]

    models = ['lr', 'tr']

    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    else:
        raise ValueError(
            'The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

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
            pool.starmap(extract_impact_weight_scales, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')