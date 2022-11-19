# Train and store Logistic Regression models on original data
import warnings
import timeit
import argparse, os
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from joblib import dump, load
from TrainMLModels import read_json, save_json, generate_model_predictions, find_optimal_thres
from TrainTreeModels import XgBoost
from TuneWeightScales import compute_weights, eval_sp
from copy import deepcopy

warnings.filterwarnings(action='ignore')

def get_sp(settings, inter_degree, y_col = 'Y'):
    train_df, features, validate_df, reweigh_method, weight_base, seed, cur_path = settings
    weights = compute_weights(train_df, reweigh_method, weight_base, alpha_g0=inter_degree, alpha_g1=inter_degree/2)
    learner = XgBoost()
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    model = learner.fit(train_data, Y_train, features, seed, weights)

    val_data = validate_df[features]
    validate_df['Y_pred_scores'] = generate_model_predictions(model, val_data)

    # optimize threshold first
    opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
    cur_thresh = opt_thres['thres']

    validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
    cur_sp = eval_sp(validate_df, 'Y_pred')
    return (model, cur_thresh, cur_sp)

def search_inter_degree(settings, degree_try, sp_previous, left=0.01, right=2.0, step=0.1):

    degree_try = degree_try or left
    best_model, best_thres, sp_try = get_sp(settings, degree_try)
    sp_previous = sp_previous or sp_try
    # print('--- degree {} sp pre {} sp cur {}'.format(degree_try, sp_previous, sp_try))

    if degree_try > right or (sp_try > 0 and sp_previous < 0):
        return (best_model, best_thres, degree_try, sp_try)
    else:
        return search_inter_degree(settings, degree_try + step, sp_try, left, right, step)


def retrain_XGB_weights(data_name, seed, reweigh_method, weight_base, input_degree=None, res_path='../intermediate/models/',
               set_suffix='S_1', data_path='data/processed/', y_col = 'Y', sensi_col='A'):

    start = timeit.default_timer()
    repo_dir = res_path.replace('intermediate/models/', '')
    cur_dir = res_path + data_name + '/'
    meta_info = read_json(repo_dir + '/' + data_path + data_name + '.json')
    n_features = meta_info['n_features']

    if reweigh_method == 'cap':
        train_df = pd.read_csv(cur_dir + '-'.join(['train', str(seed), 'bin', 'repair']) + '.csv')
    else:
        train_df = pd.read_csv(cur_dir + '-'.join(['train', str(seed), 'bin']) + '.csv')

    if reweigh_method == 'scc':
        # read the violation column for scc
        train_vio_df = pd.read_csv(cur_dir + '-'.join(['train_vio', str(seed)]) + '.csv')
        train_df['vio_cc'] = train_vio_df['vio_cc']

    validate_df = pd.read_csv(cur_dir + '-'.join(['val', str(seed), 'bin']) + '.csv')
    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed), 'bin']) + '.csv')

    if set_suffix == 'S_1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]

    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])
    val_data = validate_df[features]


    if reweigh_method == 'kam':
        weights = compute_weights(train_df, reweigh_method, weight_base)
        learner = XgBoost()
        best_model = learner.fit(train_data, Y_train, features, seed, weights)
        validate_df['Y_pred_scores'] = generate_model_predictions(best_model, val_data)
        # optimize threshold first
        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
        best_threshold = opt_thres['thres']
        best_degree = 0
    elif reweigh_method == 'cap':
        learner = XgBoost()
        best_model = learner.fit(train_data, Y_train, features, seed)
        validate_df['Y_pred_scores'] = generate_model_predictions(best_model, val_data)
        # optimize threshold first
        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
        best_threshold = opt_thres['thres']
        best_degree = 0

    elif reweigh_method == 'omn': # integrate the code from OmniFair
        best_find = 0
        if input_degree is not None: # user-specified intervention degree
            weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=input_degree)
            learner = XgBoost()
            best_model = learner.fit(train_data, Y_train, features, seed, weights)
            validate_df['Y_pred_scores'] = generate_model_predictions(best_model, val_data)

            # optimize threshold first
            opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
            best_threshold = opt_thres['thres']
            best_degree = input_degree

        else: # search the optimal intervention degree automatically
            low = 0
            high = 1
            best_acc = -1
            best_acc_unfair = -1
            best_fair = 1
            termination_flag = False
            epsilon = 0.001

            init_model = load('{}trmodel-{}-{}.joblib'.format(cur_dir, seed, set_suffix))
            validate_df['Y_pred_scores'] = generate_model_predictions(init_model, val_data)

            # optimize threshold first
            opt_thres = read_json('{}trthres-{}-{}.json'.format(cur_dir, seed, set_suffix))
            init_thresh = opt_thres['thres']

            validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > init_thresh))
            init_metric = eval_sp(validate_df, 'Y_pred')

            best_model = None
            best_degree = 0
            best_threshold = 0
            metric = init_metric

            while (abs(metric) >= epsilon or not termination_flag) and (high - low > 0.0001):
                mid = (high + low) / 2
                weights = compute_weights(train_df, reweigh_method, weight_base, omn_lam=mid)
                learner = XgBoost()
                model = learner.fit(train_data, Y_train, features, seed, weights)
                validate_df['Y_pred_scores'] = generate_model_predictions(model, val_data)
                # optimize threshold first
                opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
                acc = opt_thres['BalAcc']
                cur_thresh = opt_thres['thres']

                validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
                metric = eval_sp(validate_df, 'Y_pred')
                if (init_metric > 0 and metric < epsilon) or (init_metric < 0 and metric > -1 * epsilon):
                    high = mid
                else:
                    low = mid
                if abs(metric) <= epsilon:
                    if acc > best_acc:
                        best_acc = acc
                        best_model = deepcopy(model)
                        best_degree = mid
                        best_threshold = cur_thresh
                        best_find = 1
                        # print('+++ {} {} satisfied at {} sp {} ---'.format(data_name, seed, mid, metric))
                    if (best_acc_unfair - acc) < 0.002:
                        return
                else:
                    if best_acc_unfair < acc:
                        best_acc_unfair = acc

                    if abs(metric) < abs(best_fair):
                        best_acc = acc
                        best_model = deepcopy(model)
                        best_degree = mid
                        best_threshold = cur_thresh
                        best_fair = metric

                # print('--- degree high {} low {} mid {} acc {} sp {}---'.format(high, low, mid, acc, metric))

    elif reweigh_method == 'scc':
        if input_degree is not None: # user-specified intervention degree
            weights = compute_weights(train_df, reweigh_method, weight_base, alpha_g0=input_degree, alpha_g1=input_degree/2)
            learner = XgBoost()
            best_model = learner.fit(train_data, Y_train, features, seed, weights)
            validate_df['Y_pred_scores'] = generate_model_predictions(best_model, val_data)

            # optimize threshold first
            opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc')
            best_threshold = opt_thres['thres']
            best_degree = input_degree

        else: # search the optimal intervention degree automatically
            settings = (train_df, features, validate_df, reweigh_method, weight_base, seed, res_path)
            best_model, best_threshold, best_degree, best_recorded_sp = search_inter_degree(settings, None, None)
    else:
        raise ValueError('Not supported methods! CHOOSE FROM [scc, omn, kam].')

    end = timeit.default_timer()
    time = end - start
    if reweigh_method == 'omn':
        save_json({'time': time, 'find': best_find}, '{}trmltime-{}-{}-{}.json'.format(cur_dir, seed, reweigh_method, weight_base))
    else:
        save_json({'time': time}, '{}trmltime-{}-{}-{}.json'.format(cur_dir, seed, reweigh_method, weight_base))

    test_data = test_df[features]
    test_df['Y_pred'] = generate_model_predictions(best_model, test_data, best_threshold)

    dump(best_model, '{}trmodel-{}-{}-{}-{}.joblib'.format(cur_dir, seed, set_suffix, reweigh_method, weight_base))
    save_json({'thres': best_threshold, 'degree': best_degree}, '{}tropt-{}-{}-{}-{}.json'.format(cur_dir, seed, set_suffix, reweigh_method, weight_base))

    test_df[[sensi_col, 'Y', 'Y_pred']].to_csv('{}trtest-{}-{}-{}-{}.csv'.format(cur_dir, seed, set_suffix, reweigh_method, weight_base), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train weighted XGBoost tree models on original data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--weight", type=str, default='omn',
                        help="which weighing method to use in training ML models.")
    parser.add_argument("--base", type=str, default='one',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    parser.add_argument("--high", type=float, default=None,
                        help="scale of weights for OmniFair and SCC. Default is None for all the datasets and will be searched for optimal value automatically.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=10,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['lsac', 'cardio', 'bank', 'meps16', 'credit', 'ACSE', 'ACSP', 'ACSH', 'ACSM', 'ACSI']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

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
        else:
            datasets = datasets[:n_datasets]

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
                tasks.append([data_name, seed, args.weight, args.base, args.high, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(retrain_XGB_weights, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')