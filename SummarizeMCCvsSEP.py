import warnings
import argparse, os
from multiprocessing import Pool, cpu_count

import pandas as pd
from joblib import load
from TrainMLModels import read_json, generate_model_predictions
from EvaluateModels import assign_pred_mcc_min, assign_pred_mcc_weight, assign_pred_sensi, assign_pred_mcc_weight_group

warnings.filterwarnings('ignore')

def extract_mcc_vs_sep(data_name, seeds, models, res_path='../intermediate/models/', sim_setting='both',
                       eval_path='eval/'
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
            for group_i in range(2):
                sim_indiv = []
                group_n = test_df.query('A=={}'.format(group_i)).shape[0]
                for label_i in range(2):
                    vio_mean = cc_par['mean_train_G{}_L{}'.format(group_i, label_i)]
                    vio_col = 'vio_G{}_L{}'.format(group_i, label_i)
                    vio_min = test_df[vio_col].min()
                    if vio_mean > 0.1: # strong CC rules learned
                        # similar individuals are the ones from other groups with minimal violations
                        cur_sim_df = test_df.query('{}=={} and A=={}'.format(vio_col, vio_min, abs(1-group_i)))
                        cur_index = list(cur_sim_df.index)

                    else:
                        cur_index = []
                    if sim_setting == 'both':
                        sim_indiv = list(set(sim_indiv).intersection(set(cur_index)))
                    elif sim_setting == 'one':
                        sim_indiv = list(set(sim_indiv).union(set(cur_index)))
                    else:
                        raise ValueError('The input "similar" is not valid. CHOOSE FROM ["both", "one"].')
                key_groups.append(sim_indiv)
                base_row = [data_name, seed, test_df.shape[0], group_n,
                            '{} to {}'.format(group_i, abs(group_i - 1)), len(sim_indiv)]

                group_sim_other_df = test_df.iloc[sim_indiv, :]
                for pred_col_i, weight_set in zip(
                        ['Y_pred_min', 'Y_pred_w1', 'Y_pred_w2', 'Y_pred', 'Y_pred_A'],
                        ['MCC-MIN', 'MCC-W1', 'MCC-W2', 'ORIG', 'SEP']):
                    correct_df = group_sim_other_df.query('{} == Y'.format(pred_col_i))
                    res_df.loc[res_df.shape[0]] = base_row + [model_name, weight_set, correct_df.shape[0]]

            # simgle model cases
            weights = ['scc', 'scc', 'omn', 'kam']
            bases = ['one', 'kam', 'one', 'one']

            if model_name == 'tr':
                weights = weights + ['cap']
                bases = bases + ['one']

            for reweigh_method, weight_base in zip(weights, bases):
                scc_test_file = '{}pred-{}-{}-{}-{}.csv'.format(cur_dir, model_name, seed, reweigh_method, weight_base)

                if os.path.exists(scc_test_file):
                    test_df = pd.read_csv(scc_test_file)

                    # compute the two cases:
                    if len(key_groups) == 2:
                        for group_i, sim_indiv_list in zip([0, 1], key_groups):
                            group_n = test_df.query('A=={}'.format(group_i)).shape[0]
                            group_sim_other = test_df.iloc[sim_indiv_list, :]

                            base_row = [data_name, seed, test_df.shape[0], group_n,
                                        '{} to {}'.format(group_i, abs(group_i - 1)), group_sim_other.shape[0]]

                            correct_df = group_sim_other.query('Y_pred == Y')

                            res_df.loc[res_df.shape[0]] = base_row + [model_name, reweigh_method.upper()+'-'+weight_base.upper(), correct_df.shape[0]]
                else:
                    print('--> no', scc_test_file)

    res_df.to_csv('{}{}-{}.csv'.format(eval_path, 'min', data_name), index=False)
    print('similar res is saved at', '{}{}-{}.csv'.format(eval_path, 'min', data_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract evaluation results for similar individuals")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="extract results for all the datasets as default. Otherwise, only extract the results for the input dataset.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['', 'tr'].")
    parser.add_argument("--similar", type=str, default='both',
                        help="individuals are similar for both labels under 'all' or similar in only one label under 'one'.")

    parser.add_argument("--exec_n", type=int, default=5,
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
            tasks.append([data_name, seeds, models, res_path, args.similar])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(extract_mcc_vs_sep, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')