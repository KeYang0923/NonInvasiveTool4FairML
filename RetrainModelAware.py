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
from EvaluateModels import eval_settings
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
        val_predict = generate_model_predictions(model, val_data)
        if sum(val_predict) == 0:
            print('==> model predict only one label for val data ', data_name, model_name, seed, reweigh_method, weight_base)
        validate_df['Y_pred_scores'] = val_predict
        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', num_thresh=n_thres)
        cur_thresh = opt_thres['thres']

        validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
        cur_sp = eval_sp(validate_df, 'Y_pred')
        cur_acc = opt_thres['BalAcc']
        return (model, cur_thresh, cur_sp, cur_acc)
    else:
        print('++ no model fitted', cur_path, seed, model_name, reweigh_method, weight_base, inter_degree)
        return None


def retrain_model_with_weights_once(settings, input_degree, use_weight=True, y_col='Y'):
    learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, cur_path, cc_par = settings
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])
    if use_weight:
        weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=input_degree, alpha_g0=input_degree, alpha_g1=input_degree/2, cc_par=cc_par)
        best_model = learner.fit(train_data, Y_train, features, seed, weights)
    else:
        best_model = learner.fit(train_data, Y_train, features, seed)

    if ('bank' in cur_path or 'cardio' in cur_path):  # for a more finite search space
        n_thres = 1000
    else:
        n_thres = 100

    if best_model is not None:
        val_data = validate_df[features]
        val_predict = generate_model_predictions(best_model, val_data)
        if sum(val_predict) == 0:
            print('==> model predict only one label for val data ', data_name, model_name, seed, reweigh_method, weight_base)
        validate_df['Y_pred_scores'] = val_predict
        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', num_thresh=n_thres)
        best_threshold = opt_thres['thres']

        validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > best_threshold))
        best_sp = eval_sp(validate_df, 'Y_pred')
        best_acc = opt_thres['BalAcc']
        return (best_model, best_threshold, input_degree, best_sp, best_acc)
    else:
        print('++ no model fitted', cur_path, seed, model_name, reweigh_method, weight_base, input_degree)
        return None

def retrain_ML_with_weights_aware(data_name, seed, model_name, reweigh_method, weight_base, res_path='../intermediate/models/',
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
        degree_file = '{}par-{}-{}-{}-{}.json'.format(cur_dir, 'lr', seed, reweigh_method, weight_base)
        if os.path.exists(degree_file):
            input_degree = read_json(degree_file)['degree']
        else:
            print('--- no saved degree for', degree_file)
    else:
        train_df = pd.read_csv('{}train-cc-{}.csv'.format(cur_dir, seed))
        validate_df = pd.read_csv('{}val-cc-{}.csv'.format(cur_dir, seed))
        test_df = pd.read_csv('{}test-cc-{}.csv'.format(cur_dir, seed))
        learner = LogisticRegression()

        degree_file = '{}par-{}-{}-{}-{}.json'.format(cur_dir, 'tr', seed, reweigh_method, weight_base)
        if os.path.exists(degree_file):
            input_degree = read_json(degree_file)['degree']
        else:
            print('--- no saved degree for', degree_file)

    # print('Degree read from file', input_degree)

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

        res = retrain_model_with_weights_once(settings, input_degree)

    elif reweigh_method == 'scc':
        start = timeit.default_timer()
        cc_par = read_json('{}par-cc-{}.json'.format(cur_dir, seed))
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, seed, model_name, data_name, cc_par)
        res = retrain_model_with_weights_once(settings, input_degree)

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
        # print('++++ Best val', model_name, best_degree, best_acc, best_sp, '++++')

        res_dict = {'time': time, 'BalAcc': best_acc, 'thres': best_threshold, 'degree': best_degree, 'SPDiff': best_sp}
        save_json(res_dict, '{}par-{}-{}-{}-{}-aware.json'.format(cur_dir, model_name, seed, reweigh_method, weight_base))

        test_data = test_df[features]
        test_predict = generate_model_predictions(best_model, test_data, best_threshold)
        if sum(test_predict) == 0:
            print('==> model predict only one label for test data ', data_name, model_name, seed, reweigh_method, weight_base)
        test_df['Y_pred'] = test_predict
        # best_sp_test = eval_sp(test_df, 'Y_pred')
        # eval_res = eval_settings(test_df, sensi_col, 'Y_pred')['all']
        # print('++++ Best over test', model_name, best_degree, best_sp_test, eval_res['DI'], eval_res['BalAcc'],'++++')
        dump(best_model, '{}{}-{}-{}-{}-aware.joblib'.format(cur_dir, model_name, seed, reweigh_method, weight_base))

        test_df[[sensi_col, 'Y', 'Y_pred']].to_csv('{}pred-{}-{}-{}-{}-aware.csv'.format(cur_dir, model_name, seed, reweigh_method, weight_base), index=False)
    else:
        print('--- No model is found', data_name, seed, model_name, reweigh_method, weight_base)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train weighted ML models with weights from another model")
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
    parser.add_argument("--base", type=str, default='kam',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    # parser.add_argument("--degree", type=float, default=None,
    #                     help="additional weights in OmniFair and SCC. Default is None for all the datasets and will be searched for optimal value automatically.")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac', 'bank', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI'] #'cardio',
    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]
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
        if args.weight == 'cap':
            models = ['tr']
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
                    tasks.append([data_name, seed, model_i, args.weight, args.base, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(retrain_ML_with_weights_aware, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')