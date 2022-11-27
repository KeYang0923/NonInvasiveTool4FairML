# Simulate erroneous group membership in test datasets
# Apply fair and unfair models on these test datasets
# Evaluate the impact of the erroneous membership on the above methods using the correct membership

import warnings

import argparse, os
from multiprocessing import Pool, cpu_count
import pandas as pd
from PrepareData import read_json, save_json
from TrainMLModels import generate_model_predictions
from EvaluateModels import assign_pred_sensi, assign_pred_mcc_min, assign_pred_mcc_weight, assign_pred_mcc_weight_group, eval_settings
from joblib import load

warnings.filterwarnings('ignore')

def reverse_sensi_col(x, cand_list):
    if x.iloc[0] in cand_list:
        return int(1-x.iloc[1])
    else:
        return x.iloc[1]

def expand_core_min(err_input_df, vio_col, vio_min, vio_max, step_try, step=0.01):
    # expand the core range
    cur_unit = (vio_max-vio_min)*step
    try_min = vio_min + step_try * cur_unit
    try_sim_df = err_input_df.query('{}<={}'.format(vio_col, try_min)).copy()
    # print('>>>>>', step_try, try_min, try_sim_df.shape[0])
    return  (try_sim_df, try_min)

def simulate_error_in_sensi(input_test_df, error_ratio, vio_cols, vio_means, print_out, choose='sort', mean_thres=0.1, sensi_col='A'):
    data_name, model_name, seed, setting = print_out
    # print('===', data_name, model_name, seed, setting, '===')
    error_df = input_test_df.copy()
    test_n = error_df.shape[0]

    groups = [0, 0, 1, 1]
    labels = [0, 1, 0, 1]
    if '_' in choose:
        cur_choose_settings = choose.split('_')
        choose_group = int(cur_choose_settings[1].replace('g', ''))
        if choose_group:
            vio_cols = [vio_cols[2], vio_cols[3]]
            vio_means = [vio_means[2], vio_means[3]]
            groups = [groups[2], groups[3]]
            labels = [labels[2], labels[3]]
        else:
            vio_cols = [vio_cols[0], vio_cols[1]]
            vio_means = [vio_means[0], vio_means[1]]
            groups = [groups[0], groups[1]]
            labels = [labels[0], labels[1]]
        choose = cur_choose_settings[0]
        test_n = error_df.query('A=={}'.format(choose_group)).shape[0]
        # print('--- choose group', choose_group, test_n)
    error_n = int(error_ratio * test_n)
    min_vio_n = int(error_n / len(vio_cols))
    error_indexes = []
    extra_sample_index = []
    cur_err_n = 0
    extra_sample_data = []
    for vio_col, vio_mean, group_i, label_i in zip(vio_cols, vio_means, groups, labels):
        gl_df = error_df.query('A=={} and Y=={}'.format(group_i, label_i))
        if choose == 'min': # choose the points with the minimal violations
            vio_min = gl_df[vio_col].min()
            cur_sim_df = gl_df.query('{}=={}'.format(vio_col, vio_min))
            gl_err_n = cur_sim_df.shape[0]
            # print('---', group_i, label_i, vio_min, gl_err_n)
            if gl_err_n > min_vio_n:
                error_indexes += list(cur_sim_df.head(min_vio_n).index)
                extra_sample_index.append(1)
                cur_err_n += min_vio_n
                extra_sample_data.append(cur_sim_df.iloc[min_vio_n:, :])
            else:
                try_gl_df = cur_sim_df.copy()
                vio_max = gl_df[vio_col].max()
                step_try = 1
                while try_gl_df.shape[0] < min_vio_n and step_try < 100:
                    try_gl_df, cur_min = expand_core_min(gl_df, vio_col, vio_min, vio_max, step_try)
                    step_try += 1
                cur_sim_df = try_gl_df.copy()
                # print('---', group_i, label_i, min_vio_n, 'expand ',step_try, 'to ', cur_min, cur_sim_df.shape[0])
                error_indexes += list(cur_sim_df.index)
                extra_sample_index.append(0)
                cur_err_n += cur_sim_df.shape[0]
                extra_sample_data.append(None)
        else: # choose error_ratio points from sorted violations
            sort_gl = gl_df.sort_values(by=vio_col, ascending=True)

            # print('---', group_i, label_i, min_vio_n, sort_gl.shape[0])
            if sort_gl.shape[0] > min_vio_n:
                error_indexes += list(sort_gl.head(min_vio_n).index)
                extra_sample_index.append(1)
                cur_err_n += min_vio_n
                extra_sample_data.append(sort_gl.iloc[min_vio_n:, :])
            else:
                error_indexes += list(sort_gl.index)
                extra_sample_index.append(0)
                cur_err_n += sort_gl.shape[0]
                extra_sample_data.append(None)

    extra_case_n = sum(extra_sample_index)
    if len(error_indexes) < error_n and extra_case_n > 0: # need to add extra samples only work for group only if over all, no balance representation of group guaranteed
        need_n = error_n - cur_err_n
        extra_case_i_n = int(need_n/extra_case_n)
        for idx, extra_i, extra_data_i in zip(range(4), extra_sample_index, extra_sample_data):
            if extra_i:
                error_indexes += list(extra_data_i.head(extra_case_i_n).index)
                # print('++', idx, extra_i, len(error_indexes))
    # else:
    #     print('--- No extra available or no need to add extra!')

    chosen_err_df = error_df.iloc[error_indexes, :]
    chosen_err_df[sensi_col] = chosen_err_df[sensi_col].apply(lambda x: int(1-x))

    # error_df['ID_temp'] = error_df.index
    # error_df[sensi_col] = error_df[['ID_temp', sensi_col]].apply(lambda x: reverse_sensi_col(x, error_indexes), axis=1)
    # error_df.drop(columns=['ID_temp'], inplace=True)
    # print('---', data_name, model_name, test_n, error_n, len(error_indexes), chosen_err_df.shape[0])
    # print('===='*10, '\n\n')
    return chosen_err_df


def run_error_simulation(data_name, seed, model_name, setting, error_perc=0.2, error_core='min', res_path='../intermediate/models/',
                                data_path='data/processed/', sensi_col='A'):
    cur_dir = res_path + data_name + '/'
    error_suffix = '{}-{}'.format(error_core, error_perc)
    cc_par = read_json('{}par-cc-{}.json'.format(cur_dir, seed))
    vio_cols = ['vio_G{}_L{}'.format(group_i, label_i) for group_i in range(2) for label_i in range(2)]
    vio_means = [cc_par['mean_train_G{}_L{}'.format(group_i, label_i)] for group_i in range(2) for label_i in
                 range(2)]

    if model_name == 'tr':
        test_df = pd.read_csv('{}test-{}-bin.csv'.format(cur_dir, seed))
        vio_test_df = pd.read_csv('{}test-cc-{}.csv'.format(cur_dir, seed))
        test_df[vio_cols] = vio_test_df[vio_cols]

    else:
        test_df = pd.read_csv('{}test-cc-{}.csv'.format(cur_dir, seed))

    meta_info = read_json('{}/{}{}.json'.format(repo_dir, data_path, data_name))
    model_par = read_json('{}par-{}-{}.json'.format(cur_dir, model_name, seed))

    n_features = meta_info['n_features']  # including sensitive column
    if model_par['model_setting'] == 'S1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        raise ValueError('Error simulation only supports the feature set with sensitive attribute included!')

    # simulate errors in group membership
    print_out = (data_name, model_name, seed, setting)
    err_test_df = simulate_error_in_sensi(test_df, error_perc, vio_cols, vio_means, print_out, choose=error_core)

    err_test_data = err_test_df[features]

    if setting == 'multi':
        opt_thres = read_json('{}par-{}-{}-{}.json'.format(cur_dir, model_name, seed, setting))
        for name_suffix, cur_pred_col in zip(['', '-G0', '-G1'], ['Y_pred', 'Y_pred_G0', 'Y_pred_G1']):
            cur_model_file = '{}{}-{}{}.joblib'.format(cur_dir, model_name, seed, name_suffix)
            opt_model = load(cur_model_file)
            err_test_df[cur_pred_col] = generate_model_predictions(opt_model, err_test_data)

        err_test_df['Y_pred_A'] = err_test_df[[sensi_col, 'Y_pred_G0', 'Y_pred_G1']].apply(
            lambda x: assign_pred_sensi(x, opt_thres['sep']['thres']), axis=1)
        err_test_df['Y_pred'] = err_test_df['Y_pred'].apply(lambda x: int(x > opt_thres['orig']['thres']))

        err_test_df['Y_pred_min'] = err_test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(
            lambda x: assign_pred_mcc_min(x, vio_means), axis=1)
        err_test_df['Y_pred_w1'] = err_test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(
            lambda x: assign_pred_mcc_weight(x, vio_means), axis=1)
        err_test_df['Y_pred_w2'] = err_test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(
            lambda x: assign_pred_mcc_weight_group(x, vio_means), axis=1)


        err_test_df['Y_pred_min'] = err_test_df['Y_pred_min'].apply(lambda x: int(x > opt_thres['min']['thres']))
        err_test_df['Y_pred_w1'] = err_test_df['Y_pred_w1'].apply(lambda x: int(x > opt_thres['w1']['thres']))
        err_test_df['Y_pred_w2'] = err_test_df['Y_pred_w2'].apply(lambda x: int(x > opt_thres['w2']['thres']))

        err_test_df.to_csv('{}test-{}-{}-{}-{}.csv'.format(cur_dir, error_suffix, model_name, seed, setting), index=False)

        # reverse the group membership before evaluation
        err_test_df[sensi_col] = err_test_df[sensi_col].apply(lambda x: int(1-x))

        eval_res = {}
        for pred_y, cur_setting in zip(['Y_pred_min', 'Y_pred_w1', 'Y_pred_w2', 'Y_pred_A', 'Y_pred'],
                                       ['MCC-MIN', 'MCC-W1', 'MCC-W2', 'SEP', 'ORIG']):
            eval_res[cur_setting] = eval_settings(err_test_df, sensi_col, pred_y)

        save_json(eval_res, '{}eval-{}-{}-{}-{}.json'.format(cur_dir, error_suffix, model_name, seed, setting))


    elif setting == 'single':
        weights = ['scc', 'scc', 'omn', 'kam']
        bases = ['one', 'kam', 'one', 'one']
        if model_name == 'tr':
            weights = weights + ['cap']
            bases = bases + ['one']
        for reweigh_method, weight_base in zip(weights, bases):

            model_file = '{}{}-{}-{}-{}.joblib'.format(cur_dir, model_name, seed, reweigh_method, weight_base)
            if os.path.exists(model_file):
                opt_thres = read_json('{}par-{}-{}-{}-{}.json'.format(cur_dir, model_name, seed, reweigh_method, weight_base))
                opt_model = load(model_file)

                err_test_df['Y_pred'] = generate_model_predictions(opt_model, err_test_data, opt_thres['thres'])
                err_test_df[[sensi_col, 'Y', 'Y_pred']].to_csv('{}pred-{}-{}-{}-{}-{}.csv'.format(cur_dir, error_suffix, model_name, seed, reweigh_method, weight_base), index=False)

                # reverse the group membership before evaluation
                err_test_df[sensi_col] = err_test_df[sensi_col].apply(lambda x: int(1 - x))

                eval_res = {}
                eval_res[reweigh_method.upper()] = eval_settings(err_test_df, sensi_col, 'Y_pred')

                save_json(eval_res, '{}eval-{}-{}-{}-{}-{}.json'.format(cur_dir, error_suffix, model_name, seed, reweigh_method, weight_base))
            else:
                print('++ no model for', model_file)

    else:
        raise ValueError('Input "method" is not supported. CHOOSE FROM [multi, single].')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate Error in Group Membership and Apply Fair and Unfair Models")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")

    parser.add_argument("--setting", type=str, default='all',
                        help="which method to evaluate. CHOOSE FROM '[multi, single]'.")

    parser.add_argument("--error_p", type=float, default=0.5,
                        help="percentage of errors simulated in test set. Default is 0.5.")
    parser.add_argument("--error_core", type=str, default='min',
                        help="choose the points with minimal cc violations to simulate errors in test set. "
                             "If 'min', error_p is not valid any more. If 'sort', chooes the error_p percentage points with mininal violations."
                             "If 'min_g0', 'min_g1', 'sort_g0', or 'sort_g1' are specified, the simulation is only done on the corresponding group.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac', 'bank', 'cardio', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100,
             923]

    models = ['lr', 'tr']
    settings = ['multi', 'single']

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

    if args.setting == 'all':
        pass
    elif args.setting in settings:
        settings = [args.setting]
    else:
        raise ValueError('The input "setting" is not valid. CHOOSE FROM ["all", "multi", "single"].')

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

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds:
                for model_i in models:
                    for setting_i in settings:
                        tasks.append([data_name, seed, model_i, setting_i, args.error_p, args.error_core, res_path])
        with Pool(cpu_count() // 2) as pool:
            pool.starmap(run_error_simulation, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')