# Retrain ML models with weights
import warnings
import timeit
import argparse, os
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from joblib import dump, load
from PrepareData import read_json, save_json
from TrainMLModels import LogisticRegression, XgBoost, generate_model_predictions, find_optimal_thres
from TuneWeightScales import compute_weights, eval_sp
from copy import deepcopy

warnings.filterwarnings(action='ignore')

def get_sp(settings, inter_degree, reverse_flag=False, y_col = 'Y'):
    learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, cur_path, cc_par = settings
    weights = compute_weights(train_df, reweigh_method, weight_base, alpha_g0=inter_degree, alpha_g1=inter_degree/2,
                              cc_par=cc_par, cc_reverse=reverse_flag)
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    model = learner.fit(train_data, Y_train, features, seed, weights)

    if ('bank' in cur_path or 'cardio' in cur_path) and model_name == 'tr':  # for a more finite search space
        n_thres = 1000
    else:
        n_thres = 100

    if model is not None:
        val_data = validate_df[features]
        validate_df['Y_pred_scores'] = generate_model_predictions(model, val_data)

        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', num_thresh=n_thres)
        cur_thresh = opt_thres['thres']

        validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
        cur_sp = eval_sp(validate_df, 'Y_pred')
        cur_acc = opt_thres['BalAcc']
        return (model, cur_thresh, cur_sp, cur_acc)
    else:
        print('++ no model fitted', cur_path, seed, model_name, reweigh_method, weight_base, inter_degree)
        return None

def search_inter_degree_scc(settings, degree_try, sp_pre, reverse_try, left=0.01, right=2.0, step=0.1, sp_diff_bound=0.02):

    degree_try = degree_try or left
    reverse_try = reverse_try or False
    sp_pre = sp_pre or 1
    res_try = get_sp(settings, degree_try, reverse_try)
    if res_try is not None:
        model_try, thres_try, sp_try, acc_try = res_try
        # print('--- {} {} {} {} ---'.format(settings[7], degree_try, acc_try, sp_try))

        if sp_try > 0 and settings[4] == 'scc' and settings[5] == 'one':
            reverse_try = True

        if degree_try > right or (sp_try >= 0 and sp_try < sp_diff_bound) or abs(abs(sp_try) - abs(sp_pre)) < 0.001:
            return (model_try, thres_try, degree_try, sp_try, acc_try)
        else:
            return search_inter_degree_scc(settings, degree_try + step, sp_try, reverse_try, left, right, step)
    else:
        print('++ no model found', settings[6], settings[7], settings[4], settings[5], degree_try)
        return None

def search_inter_degree_omn(settings, low=0, high=2, epsilon = 0.02, y_col='Y'):
    learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, cur_path, _ = settings

    best_acc = -1
    best_sp = 1
    termination_flag = False

    init_model = load('{}{}-{}.joblib'.format(cur_path, model_name, seed))
    init_thresh = read_json('{}par-{}-{}.json'.format(cur_path, model_name, seed))['thres']
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    val_data = validate_df[features]
    validate_df['Y_pred_scores'] = generate_model_predictions(init_model, val_data)
    validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > init_thresh))
    init_sp = eval_sp(validate_df, 'Y_pred')

    best_model = None
    best_degree = 0
    best_threshold = 0

    cur_sp = init_sp
    if ('bank' in cur_path or 'cardio' in cur_path) and model_name == 'tr':  # for a more finite search space
        n_thres = 1000
    else:
        n_thres = 100

    while (abs(cur_sp) >= epsilon or not termination_flag) and (high - low > 0.0001):
        mid = (high + low) / 2
        weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=mid)
        cur_model = learner.fit(train_data, Y_train, features, seed, weights)
        if cur_model is not None:
            validate_df['Y_pred_mid'] = generate_model_predictions(cur_model, val_data)
            model_prob = validate_df['Y_pred_mid'].unique()
            if len(model_prob) > 5: # no reasonable probability outputted
                opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', pred_col='Y_pred_mid', num_thresh=n_thres)
                cur_acc = opt_thres['BalAcc']
                cur_thresh = opt_thres['thres']

                validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
                cur_sp = eval_sp(validate_df, 'Y_pred')

                # print('--- {} {} {} {}---'.format(model_name, mid, cur_acc, cur_sp))
            else:
                print('++ no model fitted', seed, model_name, reweigh_method, weight_base, mid)
                cur_sp = 1
                cur_acc = -1
        else:
            print('++ no model fitted', seed, model_name, reweigh_method, weight_base, mid)
            cur_sp = 1
            cur_acc = -1

        if (init_sp > 0 and cur_sp < epsilon) or (init_sp < 0 and cur_sp > -1 * epsilon):
            high = mid
        else:
            low = mid

        if abs(cur_sp) <= epsilon:
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_sp = cur_sp
                best_model = deepcopy(cur_model)
                best_degree = mid
                best_threshold = cur_thresh
                print('---> OMN {} {} satisfied at degree {} with sp {} acc {} ---'.format(model_name, seed, mid, best_sp, best_acc))
                break
        else:
            if abs(cur_sp) < abs(best_sp):
                best_acc = cur_acc
                best_sp = cur_sp
                best_model = deepcopy(cur_model)
                best_degree = mid
                best_threshold = cur_thresh

    if best_threshold == 0: # no model fitted case
        return None
    else:
        return (best_model, best_threshold, best_degree, best_sp, best_acc)

def retrain_model_with_weights_once(settings, input_degree, use_weight=True, y_col='Y'):
    learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, cur_path, cc_par = settings
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])
    if use_weight:
        weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=input_degree, alpha_g0=input_degree, alpha_g1=input_degree/2, cc_par=cc_par)
        best_model = learner.fit(train_data, Y_train, features, seed, weights)
    else:
        best_model = learner.fit(train_data, Y_train, features, seed)

    if ('bank' in cur_path or 'cardio' in cur_path) and model_name == 'tr':  # for a more finite search space
        n_thres = 1000
    else:
        n_thres = 100

    if best_model is not None:
        val_data = validate_df[features]
        validate_df['Y_pred_scores'] = generate_model_predictions(best_model, val_data)
        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', num_thresh=n_thres)
        best_threshold = opt_thres['thres']

        validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > best_threshold))
        best_sp = eval_sp(validate_df, 'Y_pred')
        best_acc = opt_thres['BalAcc']
        return (best_model, best_threshold, input_degree, best_sp, best_acc)
    else:
        print('++ no model fitted', cur_path, seed, model_name, reweigh_method, weight_base, input_degree)
        return None

def retrain_ML_with_weights(data_name, seed, model_name, reweigh_method, weight_base, input_degree=None, res_path='../intermediate/models/',
                            data_path='data/processed/', sensi_col='A'):

    repo_dir = res_path.replace('intermediate/models/', '')
    cur_dir = res_path + data_name + '/'

    if model_name == 'tr':
        if reweigh_method == 'cap':
            train_df = pd.read_csv('{}train-{}-bin-{}.csv'.format(cur_dir, seed, reweigh_method))
        else:
            train_df = pd.read_csv('{}train-{}-bin.csv'.format(cur_dir, seed))

        validate_df = pd.read_csv('{}val-{}-bin.csv'.format(cur_dir, seed))
        test_df = pd.read_csv('{}test-{}-bin.csv'.format(cur_dir, seed))

        vio_train_df = pd.read_csv('{}train-cc-{}.csv'.format(cur_dir, seed))
        train_df['vio_cc'] = vio_train_df['vio_cc']
        learner = XgBoost()
    else:
        train_df = pd.read_csv('{}train-cc-{}.csv'.format(cur_dir, seed))
        validate_df = pd.read_csv('{}val-cc-{}.csv'.format(cur_dir, seed))
        test_df = pd.read_csv('{}test-cc-{}.csv'.format(cur_dir, seed))
        learner = LogisticRegression()

    meta_info = read_json('{}/{}{}.json'.format(repo_dir, data_path, data_name))
    feature_setting = read_json('{}par-{}-{}.json'.format(cur_dir, model_name, seed))['model_setting']
    n_features = meta_info['n_features']  # including sensitive column

    if feature_setting == 'S1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]

    if reweigh_method == 'kam':
        start = timeit.default_timer()
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, data_name, None)
        res = retrain_model_with_weights_once(settings, 0)

    elif reweigh_method == 'omn': # integrate the code from OmniFair
        start = timeit.default_timer()
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, cur_dir, None)

        if input_degree:# user-specified intervention degree
            res = retrain_model_with_weights_once(settings, input_degree)
        else: # search the optimal intervention degree automatically
            res = search_inter_degree_omn(settings)


    elif reweigh_method == 'scc':
        start = timeit.default_timer()
        cc_par = read_json('{}par-cc-{}.json'.format(cur_dir, seed))
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, data_name, cc_par)
        if input_degree: # user-specified intervention degree
            res = retrain_model_with_weights_once(settings, input_degree)
        else: # search the optimal intervention degree automatically
            if data_name == 'cardio':
                cur_left = 11
                cur_right = 12
                cur_step = 0.1
            else:
                cur_left = 0.01
                cur_right = 2
                cur_step = 0.01
            res = search_inter_degree_scc(settings, None, None, None, left=cur_left, right=cur_right, step=cur_step)

    elif reweigh_method == 'cap':
        start = timeit.default_timer()
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, data_name, None)
        res = retrain_model_with_weights_once(settings, 0, use_weight=False)
    else:
        raise ValueError('Not supported methods! CHOOSE FROM [scc, omn, kam, cap].')

    end = timeit.default_timer()
    time = end - start

    if res is not None:
        best_model, best_threshold, best_degree, best_sp, best_acc = res
        # print('++++ Best', model_name, best_degree, best_acc, best_sp, '++++')

        res_dict = {'time': time, 'BalAcc': best_acc, 'thres': best_threshold, 'degree': best_degree, 'SPDiff': best_sp}
        save_json(res_dict, '{}par-{}-{}-{}-{}.json'.format(cur_dir, model_name, seed, reweigh_method, weight_base))

        test_data = test_df[features]
        test_df['Y_pred'] = generate_model_predictions(best_model, test_data, best_threshold)

        dump(best_model, '{}{}-{}-{}-{}.joblib'.format(cur_dir, model_name, seed, reweigh_method, weight_base))

        test_df[[sensi_col, 'Y', 'Y_pred']].to_csv('{}pred-{}-{}-{}-{}.csv'.format(cur_dir, model_name, seed, reweigh_method, weight_base), index=False)
    else:
        print('--- No model is found', data_name, seed, model_name, reweigh_method, weight_base)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train weighted ML models")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")

    parser.add_argument("--weight", type=str, default='scc',
                        help="which weighing method to use in training ML models.")
    parser.add_argument("--base", type=str, default='one',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    parser.add_argument("--degree", type=float, default=None,
                        help="additional weights in OmniFair and SCC. Default is None for all the datasets and will be searched for optimal value automatically.")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac', 'bank', 'cardio', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100,
             923]
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
                    tasks.append([data_name, seed, model_i, args.weight, args.base, args.degree, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(retrain_ML_with_weights, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')