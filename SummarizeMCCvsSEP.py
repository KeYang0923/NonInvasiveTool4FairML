import warnings
import argparse, os
from multiprocessing import Pool, cpu_count

import pandas as pd
from PrepareData import read_json, save_json
from EvaluateModels import eval_settings

warnings.filterwarnings('ignore')

def eval_min_violation(data_name, seeds, models, res_path='../intermediate/models/', label=None,
                       eval_path='eval/', sensi_col='A'
                       ):
    repo_dir = res_path.replace('intermediate/models/', '')
    eval_path = repo_dir + eval_path

    cur_dir = res_path + data_name + '/'
    res_df = pd.DataFrame(columns=['data', 'seed', 'test_n', 'group_n', 'sim', 'sim_n', 'model', 'method', 'correct_n'])

    for seed in seeds:
        cc_par = read_json('{}par-cc-{}.json'.format(cur_dir, seed))
        for model_name in models:
            # multi model case
            test_df = pd.read_csv('{}/test-{}-{}-multi.csv'.format(cur_dir, model_name, seed))
            key_groups = []
            name_suffix = []
            for group_i in range(2):
                if label is None:
                    labels = range(2)
                    eval_suffix = 'sim-both-{}'.format(group_i)
                elif label == 'pos':
                    labels = [1]
                    eval_suffix = 'sim-pos-{}'.format(group_i)
                elif label == 'neg':
                    labels = [0]
                    eval_suffix = 'sim-neg-{}'.format(group_i)
                else:
                    raise ValueError('The "label" is need to be in [None, pos, neg].')
                name_suffix.append(eval_suffix)
                sim_indiv = []
                group_n = test_df.query('A=={}'.format(group_i)).shape[0]
                for label_i in labels:
                    vio_mean = cc_par['mean_train_G{}_L{}'.format(group_i, label_i)]
                    vio_col = 'vio_G{}_L{}'.format(group_i, label_i)
                    vio_min = test_df[vio_col].min()
                    if vio_mean > 0.1: # strong CC rules learned
                        # similar individuals are the ones from other groups with minimal violations
                        cur_sim_df = test_df.query('{}=={} and A=={}'.format(vio_col, vio_min, group_i))
                        cur_index = list(cur_sim_df.index)

                    else:
                        cur_index = []
                    sim_indiv = list(set(sim_indiv).union(set(cur_index)))

                key_groups.append(sim_indiv)
                base_row = [data_name, seed, test_df.shape[0], group_n,
                            '{} to {}'.format(group_i, abs(group_i - 1)), len(sim_indiv)]

                group_sim_other_df = test_df.iloc[sim_indiv, :]
                eval_res = {}
                for pred_col_i, weight_set in zip(
                        ['Y_pred_min', 'Y_pred_w1', 'Y_pred_w2', 'Y_pred', 'Y_pred_A'],
                        ['MCC-MIN', 'MCC-W1', 'MCC-W2', 'ORIG', 'SEP']):
                    correct_df = group_sim_other_df.query('{} == Y'.format(pred_col_i))
                    eval_res[weight_set] = eval_settings(group_sim_other_df, sensi_col, pred_col_i)

                    res_df.loc[res_df.shape[0]] = base_row + [model_name, weight_set, correct_df.shape[0]]

                save_json(eval_res, '{}eval-{}-{}-{}-{}.json'.format(cur_dir, eval_suffix, model_name, seed, 'multi'))

            # simgle model cases
            weights = ['scc', 'scc', 'omn', 'kam']
            bases = ['one', 'kam', 'one', 'one']

            if model_name == 'tr':
                weights = weights + ['cap']
                bases = bases + ['one']

            for reweight_method, weight_base in zip(weights, bases):
                scc_test_file = '{}pred-{}-{}-{}-{}.csv'.format(cur_dir, model_name, seed, reweight_method, weight_base)

                if os.path.exists(scc_test_file):
                    test_df = pd.read_csv(scc_test_file)
                    cur_single = reweight_method.upper()+'-'+weight_base.upper()
                    # compute the two cases:
                    if len(key_groups) == 2:
                        for group_i, sim_indiv_list, eval_suffix in zip([0, 1], key_groups, name_suffix):
                            group_n = test_df.query('A=={}'.format(group_i)).shape[0]
                            group_sim_other = test_df.iloc[sim_indiv_list, :]
                            eval_res = {}
                            eval_res[reweight_method.upper()] = eval_settings(group_sim_other, sensi_col, 'Y_pred')

                            save_json(eval_res, '{}eval-{}-{}-{}-{}-{}.json'.format(cur_dir, eval_suffix, model_name, seed, reweight_method, weight_base))

                            base_row = [data_name, seed, test_df.shape[0], group_n,
                                        '{} to {}'.format(group_i, abs(group_i - 1)), group_sim_other.shape[0]]

                            correct_df = group_sim_other.query('Y_pred == Y')

                            res_df.loc[res_df.shape[0]] = base_row + [model_name, cur_single, correct_df.shape[0]]
                else:
                    print('--> no', scc_test_file)

    if label is None:
        eval_suffix = 'sim-both-{}'.format(group_i)
    elif label == 'pos':
        eval_suffix = 'sim-pos-{}'.format(group_i)
    elif label == 'neg':
        eval_suffix = 'sim-neg-{}'.format(group_i)
    else:
        raise ValueError('The "label" is need to be in [None, pos, neg].')
    res_df.to_csv('{}n-{}-{}.csv'.format(eval_path, eval_suffix, data_name), index=False)
    print('similar res is saved at', '{}n-{}-{}.csv'.format(eval_path, eval_suffix, data_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract evaluation results for similar individuals")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="extract results for all the datasets as default. Otherwise, only extract the results for the input dataset.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")
    parser.add_argument("--focus", type=str, default=None,
                        help="focus on only positive or negative label by specifying 'pos' or 'neg'. On both labels by specifying as None.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100,
             923]
    models = ['lr', 'tr']

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
            tasks.append([data_name, seeds, models, res_path, args.focus])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(eval_min_violation, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')