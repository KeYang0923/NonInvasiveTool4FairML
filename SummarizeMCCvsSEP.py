import warnings
import argparse, os
from multiprocessing import Pool, cpu_count

import pandas as pd
from joblib import load
from TrainMLModels import read_json, generate_model_predictions
from EvaluateModels import assign_pred_mcc_min, assign_pred_mcc_weight, assign_pred_sensi, assign_pred_mcc_weight_group

warnings.filterwarnings('ignore')

def extract_mcc_vs_sep(data_name, seeds, models, res_path='../intermediate/models/',
                       method='mcc', eval_path='eval/',
                       set_suffix = 'S_1', y_col='Y', sensi_col='A', n_groups=2):
    repo_dir = res_path.replace('intermediate/models/', '')
    eval_path = repo_dir + eval_path

    cur_dir = res_path + data_name + '/'
    res_df = pd.DataFrame(columns=['data', 'model', 'seed', 'case', 'case_n', 'mcc_weight', 'correct_perc'])

    vio_cols = ['vio_G{}_L{}'.format(group_i, label_i) for group_i in range(n_groups) for label_i in range(2)]
    sep_pred_col = 'Y_pred_{}'.format(sensi_col)

    for seed in seeds:
        for model in models:
            if model == 'lr':
                model = ''
                test_df = pd.read_csv(cur_dir + '-'.join(['test_vio', str(seed)]) + '.csv')
            else: # model == 'tr':
                test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed), 'bin']) + '.csv')

                vio_test_df = pd.read_csv(cur_dir + '-'.join(['test_vio', str(seed)]) + '.csv')
                test_df[vio_cols] = vio_test_df[vio_cols]

            if set_suffix == 'S_1':
                features = [x for x in test_df.columns if x != y_col]
            else:
                features = [x for x in test_df.columns if x != y_col and x != sensi_col]

            test_data = test_df[features]

            model_file = '{}{}model-{}-{}.joblib'.format(cur_dir, model, seed, set_suffix)

            if os.path.exists(model_file):
                thres_all = read_json('{}{}evalthres-{}-{}-{}.json'.format(cur_dir, model, seed, set_suffix, method))

                for group_suffix in [None, 'G0', 'G1']:
                    if group_suffix is None:
                        opt_model = load(model_file)
                        cur_pred_col = 'Y_pred'
                    else:
                        opt_model = load('{}{}model-{}-{}-{}.joblib'.format(cur_dir, model, seed, group_suffix, set_suffix))
                        cur_pred_col = 'Y_pred_' + group_suffix

                    test_df[cur_pred_col] = generate_model_predictions(opt_model, test_data)

                orig_opt_thres = thres_all['orig']['thres']
                min_opt_thres = thres_all['min']['thres']
                w1_opt_thres = thres_all['w1']['thres']
                w2_opt_thres = thres_all['w2']['thres']
                a_opt_thres = thres_all['sep']['thres']

                test_df['Y_pred_min'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_min(x), axis=1)
                test_df['Y_pred_w1'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight(x), axis=1)
                test_df['Y_pred_w2'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight_group(x), axis=1)
                test_df[sep_pred_col] = test_df[[sensi_col, 'Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_sensi(x), axis=1)

                test_df['Y_pred_min'] = test_df['Y_pred_min'].apply(lambda x: int(x > min_opt_thres))
                test_df['Y_pred_w1'] = test_df['Y_pred_w1'].apply(lambda x: int(x > w1_opt_thres))
                test_df['Y_pred_w2'] = test_df['Y_pred_w2'].apply(lambda x: int(x > w2_opt_thres))
                test_df[sep_pred_col] = test_df['Y_pred_A'].apply(lambda x: int(x > a_opt_thres))

                test_df['Y_pred'] = test_df['Y_pred'].apply(lambda x: int(x > orig_opt_thres['thres']))

                test_df.to_csv('{}{}test-{}-{}-{}.csv'.format(cur_dir, model, seed, set_suffix, method), index=False)
                if model == '':
                    model_name = 'LR'
                else:
                    model_name = 'TR'
                # compute the two cases:
                for group_i in range(n_groups):
                    err_df_a = test_df.query('Y != {} and {}=={}'.format(sep_pred_col, sensi_col, group_i))
                    group_sim_other = err_df_a.query('vio_G{}_L0 > vio_G{}_L0 and vio_G{}_L1 > vio_G{}_L1'.format(group_i, abs(group_i-1), group_i, abs(group_i-1)))

                    for pred_col_i, weight_set in zip(['Y_pred_min', 'Y_pred_w1', 'Y_pred_w2', 'Y_pred'], ['MIN', 'W1', 'W2', 'ORIG']):
                        pred_plus_df = group_sim_other.query('{} == Y'.format(pred_col_i))
                        ratio = round(pred_plus_df.shape[0]/err_df_a.shape[0],3)
                        res_df.loc[res_df.shape[0]] = [data_name, model_name, seed, '{} to {}'.format(group_i, abs(group_i-1)), err_df_a.shape[0], weight_set, ratio]
            else:
                print('--> no', model_file)

    res_df.to_csv('{}{}-{}.csv'.format(eval_path, method, data_name), index=False)

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
            pool.starmap(extract_mcc_vs_sep, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')