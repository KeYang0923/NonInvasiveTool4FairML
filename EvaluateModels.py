import warnings
import argparse, os
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from PrepareData import read_json, save_json
from TrainMLModels import generate_model_predictions, find_optimal_thres

warnings.filterwarnings('ignore')

def eval_predictions(y_true, y_pred, label_order=[0, 1], metrics=['AUC', 'ACC', 'ERR', 'FPR', 'FNR', 'PR', 'TPR', 'TNR', 'TP', 'FN', 'TN', 'FP', 'SR']):

    ACC = round(accuracy_score(y_true, y_pred), 3)
    try:
        AUC = round(roc_auc_score(y_true, y_pred), 3)
    except Exception:
        AUC = -1

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=label_order).ravel()
    pred_P = TP+FP
    # pred_N = TN+FN
    P = TP + FN
    N = TN + FP
    eval_dict = dict(
                PR = P/ (P+N) if (P+N) > 0.0 else np.float64(0.0),
                P = np.float64(P), N = np.float64(N), FP = np.float64(FP), TP = np.float64(TP), FN = np.float64(FN), TN = np.float64(TN),
                TPR=TP / P if P > 0.0 else np.float64(0.0),
                TNR=TN / N if N > 0.0 else np.float64(0.0),
                FPR=FP / N if N > 0.0 else np.float64(0.0),
                FNR=FN / P if P > 0.0 else np.float64(0.0),
                PPV=TP / (TP+FP) if (TP+FP) > 0.0 else np.float64(0.0),
                NPV=TN / (TN+FN) if (TN+FN) > 0.0 else np.float64(0.0),
                FDR=FP / (FP+TP) if (FP+TP) > 0.0 else np.float64(0.0),
                FOR=FN / (FN+TN) if (FN+TN) > 0.0 else np.float64(0.0),
                ACC=ACC,
                AUC=AUC if AUC != -1 else np.float64(0.0),
                ERR=1-ACC,
                F1=2*TP / (2*TP+FP+FN) if (2*TP+FP+FN) > 0.0 else np.float64(0.0),
                SR= pred_P / len(y_pred)
            )
    return {key:eval_dict[key] for key in metrics}


def eval_settings(test_eval_df, sensi_col, pred_col, n_groups=2):

    results = {}
    PR_all = []
    TPR_all = []
    FPR_all = []
    FNR_all = []
    ERR_all = []

    for group_i in range(n_groups):
        group_df = test_eval_df[test_eval_df[sensi_col] == group_i]
        eval_res = eval_predictions(group_df['Y'], group_df[pred_col])
        bal_acc = 0.5 * (eval_res['TPR'] + eval_res['TNR'])
        eval_res.update({'BalAcc': bal_acc})
        results['G'+str(group_i)] = eval_res

        PR_all.append(eval_res['SR'])
        TPR_all.append(eval_res['TPR'])
        FPR_all.append(eval_res['FPR'])
        FNR_all.append(eval_res['FNR'])
        ERR_all.append(eval_res['ERR'])

    eval_res_all = eval_predictions(test_eval_df['Y'], test_eval_df[pred_col])
    bal_acc_all = 0.5 * (eval_res_all['TPR'] + eval_res_all['TNR'])
    # additional overall fairness measures
    di_pr = PR_all[0] / PR_all[1] if PR_all[1] > 0 else 0
    eq = TPR_all[0] - TPR_all[1]
    fpr_diff = FPR_all[0] - FPR_all[1]
    fnr_diff = FNR_all[0] - FNR_all[1]
    err_diff = ERR_all[0] - ERR_all[1]

    avg_odds = 0.5 * (FPR_all[0] - FPR_all[1] + eq)
    sp_diff = PR_all[0] - PR_all[1]
    for metric_i, value_i in zip(['BalAcc', 'DI', 'EQDiff', 'AvgOddsDiff', 'SPDiff', 'FPRDiff', 'FNRDiff', 'ERRDiff'], [bal_acc_all, di_pr, eq, avg_odds, sp_diff, fpr_diff, fnr_diff, err_diff]):
        eval_res_all.update({metric_i: value_i})
    results['all'] = eval_res_all
    return results

def assign_pred_mcc_min(x, vio_means, n_vio=4): # assign prediction based on minimal violation
    violations = [x.iloc[i]* (1-mean_i) for i, mean_i in enumerate(vio_means)]

    pred_index = violations.index(min(violations))
    if pred_index <= 1:
        return x.iloc[n_vio]
    else:
        return x.iloc[n_vio+1]

def assign_pred_mcc_weight(x, vio_means, n_vio=4): # assign prediction based on minimal violation
    # vio = [1- x.iloc[i] for i in range(n_vio)]
    weights = []
    for i, mean_i in enumerate(vio_means):
        if mean_i > 0.1:
            weights.append(1-x.iloc[i])
        else:
            weights.append(0)

    weights_norm = sum(weights)
    if weights_norm == 0:
        weights_norm = 0.0001

    return weights[0]/weights_norm * (1- x.iloc[n_vio]) + weights[1]/weights_norm * x.iloc[n_vio] + weights[2]/weights_norm * (1- x.iloc[n_vio+1]) + weights[3]/weights_norm * x.iloc[n_vio+1]

def assign_pred_mcc_weight_group(x, vio_means, n_vio=4): # assign prediction based on minimal violation
    weights = []
    for i, mean_i in enumerate(vio_means):
        if mean_i > 0.1:
            weights.append(1-x.iloc[i])
        else:
            weights.append(0)

    if weights[2] > weights[3]: # use negative probability for group 1
        weight_g1 = weights[2]
        prob_g1 = 1- x.iloc[n_vio+1]
    else:
        weight_g1 = weights[3]
        prob_g1 = x.iloc[n_vio + 1]

    if weights[0] > weights[1]: # use negative probability for group 1
        weight_g0 = weights[0]
        prob_g0 = 1- x.iloc[n_vio]
    else:
        weight_g0 = weights[1]
        prob_g0 = x.iloc[n_vio]

    # weight_g1 = 1- min(x.iloc[2], x.iloc[3])
    # weight_g0 = 1- min(x.iloc[0], x.iloc[1])
    weights_norm = weight_g1 + weight_g0
    if weights_norm == 0:
        weights_norm = 0.0001
    return weight_g0/weights_norm * prob_g0 + weight_g1/weights_norm * prob_g1

def assign_pred_sensi(x, thres_groups): # assign prediction based on the group membership of sensitive attribute
    if x.iloc[0]: # sensi_col == 1, majority group
        return int(x.iloc[2] > thres_groups[1])
    else:
        return int(x.iloc[1] > thres_groups[0])

def eval_predicitons(data_name, seed, model_name, setting,
                     res_path='../intermediate/models/', cc_opt=True,
                     model_aware=False,
                     data_path='data/processed/', sensi_col='A'):
    cur_dir = res_path + data_name + '/'
    if cc_opt:
        opt_suffix = ''
    else:
        opt_suffix = '-noOPT'

    if setting == 'multi':
        cc_par = read_json('{}par-cc-{}{}.json'.format(cur_dir, seed, opt_suffix))
        vio_cols = ['vio_G{}_L{}'.format(group_i, label_i) for group_i in range(2) for label_i in range(2)]
        vio_means = [cc_par['mean_train_G{}_L{}'.format(group_i, label_i)] for group_i in range(2) for label_i in range(2)]

        if model_name == 'tr':
            test_df = pd.read_csv('{}test-{}-bin.csv'.format(cur_dir, seed))
            val_df = pd.read_csv('{}val-{}-bin.csv'.format(cur_dir, seed))

            vio_test_df = pd.read_csv('{}test-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))
            vio_val_df = pd.read_csv('{}val-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))

            test_df[vio_cols] = vio_test_df[vio_cols]
            val_df[vio_cols] = vio_val_df[vio_cols]
        else:
            val_df = pd.read_csv('{}val-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))
            test_df = pd.read_csv('{}test-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))

        meta_info = read_json('{}/{}{}.json'.format(repo_dir, data_path, data_name))
        model_par = read_json('{}par-{}-{}.json'.format(cur_dir, model_name, seed))
        n_features = meta_info['n_features']  # including sensitive column

        if model_par['model_setting'] == 'S1':
            features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
        else:
            features = ['X{}'.format(i) for i in range(1, n_features)]

        test_data = test_df[features]
        val_data = val_df[features]


        for name_suffix, cur_pred_col in zip(['', '-G0', '-G1'], ['Y_pred', 'Y_pred_G0', 'Y_pred_G1']):
            cur_model_file = '{}{}-{}{}.joblib'.format(cur_dir, model_name, seed, name_suffix)
            opt_model = load(cur_model_file)

            test_predict = generate_model_predictions(opt_model, test_data)
            if sum(test_predict) == 0:
                print('==> model predict only one label for test data in MCC ', cur_pred_col, data_name, model_name, seed)
            test_df[cur_pred_col] = test_predict
            val_predict = generate_model_predictions(opt_model, val_data)
            if sum(val_predict) == 0:
                print('==> model predict only one label for val data in MCC ', cur_pred_col, data_name, model_name, seed)
            val_df[cur_pred_col] = val_predict

        test_df['Y_pred_A'] = test_df[[sensi_col, 'Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_sensi(x, [model_par['thres_g0'], model_par['thres_g1']]), axis=1)
        test_df['Y_pred'] = test_df['Y_pred'].apply(lambda x: int(x > model_par['thres']))

        if sum(vio_means) < 0.1: # no reasonable means are produced for all the four subsets
            pass
        else:
            # find the optimal threshold over validate data for mcc weighted version
            val_df['Y_pred_min'] = val_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_min(x, vio_means), axis=1)
            val_df['Y_pred_w1'] = val_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight(x, vio_means), axis=1)
            val_df['Y_pred_w2'] = val_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight_group(x, vio_means), axis=1)

            min_opt_thres = find_optimal_thres(val_df, opt_obj='BalAcc', pred_col='Y_pred_min')
            w1_opt_thres = find_optimal_thres(val_df, opt_obj='BalAcc', pred_col='Y_pred_w1')
            w2_opt_thres = find_optimal_thres(val_df, opt_obj='BalAcc', pred_col='Y_pred_w2')

            test_df['Y_pred_min'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_min(x, vio_means), axis=1)
            test_df['Y_pred_w1'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight(x, vio_means), axis=1)
            test_df['Y_pred_w2'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight_group(x, vio_means), axis=1)

            test_df['Y_pred_min'] = test_df['Y_pred_min'].apply(lambda x: int(x > min_opt_thres['thres']))
            test_df['Y_pred_w1'] = test_df['Y_pred_w1'].apply(lambda x: int(x > w1_opt_thres['thres']))
            test_df['Y_pred_w2'] = test_df['Y_pred_w2'].apply(lambda x: int(x > w2_opt_thres['thres']))

            test_df.to_csv('{}test-{}-{}-{}{}.csv'.format(cur_dir, model_name, seed, setting, opt_suffix), index=False)

            eval_res = {}
            for pred_y, cur_setting in zip(['Y_pred_min', 'Y_pred_w1', 'Y_pred_w2', 'Y_pred_A', 'Y_pred'], ['MCC-MIN', 'MCC-W1', 'MCC-W2', 'SEP', 'ORIG']):
                eval_res[cur_setting] = eval_settings(test_df, sensi_col, pred_y)

            save_json(eval_res, '{}eval-{}-{}-{}{}.json'.format(cur_dir, model_name, seed, setting, opt_suffix))

            par_dict = {'orig': {'thres': model_par['thres'], 'BalAcc': model_par['BalAcc']},
                        'min': {'thres': min_opt_thres['thres'], 'BalAcc': min_opt_thres['BalAcc']},
                        'w1': {'thres': w1_opt_thres['thres'], 'BalAcc': w1_opt_thres['BalAcc']},
                        'w2': {'thres': w2_opt_thres['thres'], 'BalAcc': w2_opt_thres['BalAcc']},
                        'sep': {'thres': (model_par['thres_g0'], model_par['thres_g1']),
                                'BalAcc': (model_par['BalAcc_g0'], model_par['BalAcc_g1'])},
                        }

            save_json(par_dict, '{}par-{}-{}-{}{}.json'.format(cur_dir, model_name, seed, setting, opt_suffix))

    elif setting == 'single':
        if 'syn' in data_name:
            # for synthetic data
            weights = ['scc']
            bases = ['kam']
        else:
            if not cc_opt:
                # no optimization of CC
                weights = ['scc']
                bases = ['kam' + opt_suffix]
            else: # default optimization
                if not model_aware: # for experiments with right weights
                    weights = ['scc', 'scc', 'omn', 'kam']
                    bases = ['one', 'kam', 'one', 'one']
                    if model_name == 'tr':
                        weights = weights + ['cap']
                        bases = bases + ['one']
                else: # for experiments with model-aware weights
                    weights = ['scc', 'omn']
                    bases = ['kam-aware', 'one-aware']

        for reweight_method, weight_base in zip(weights, bases):
            eval_res = {}
            test_file = '{}pred-{}-{}-{}-{}.csv'.format(cur_dir, model_name, seed, reweight_method, weight_base)
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file)
                eval_res[reweight_method.upper()] = eval_settings(test_df, sensi_col, 'Y_pred')
                save_json(eval_res, '{}eval-{}-{}-{}-{}.json'.format(cur_dir, model_name, seed, reweight_method, weight_base))

            else:
                print('++ no model for', test_file)

    else:
        raise ValueError('Input "method" is not supported. CHOOSE FROM [multi, single].')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval fairness interventions on real data")
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
    parser.add_argument("--exec_n", type=int, default=10,
                        help="number of executions with different random seeds. Default is 20.")

    parser.add_argument("--opt", type=int, default=0,
                        help="whether to apply the optimization for CC tool.")

    parser.add_argument("--aware", type=int, default=0,
                        help="whether to evaluate for the model aware weights.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac', 'bank', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI'] #'cardio',

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]
    # seeds = [88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]

    models = ['lr', 'tr']
    settings = ['multi', 'single']

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    elif 'syn' in args.data:
        datasets = ['syn{}'.format(x) for x in seeds]
    else:
        raise ValueError(
            'The input "data" is not valid. CHOOSE FROM ["syn", "lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

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



    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds:
                for model_i in models:
                    for setting_i in settings:
                        tasks.append([data_name, seed, model_i, setting_i, res_path, args.opt, args.aware])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(eval_predicitons, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')