import warnings
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from TrainMLModels import read_json, generate_model_predictions, save_json

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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

def assign_pred_mcc_min(x, n_vio=4): # assign prediction based on minimal violation
    violations = [x.iloc[i] for i in range(n_vio)]
    pred_index = violations.index(min(violations))
    if pred_index <= 1:
        return x.iloc[n_vio]
    else:
        return x.iloc[n_vio+1]

def assign_pred_mcc_weight(x, n_vio=4): # assign prediction based on minimal violation
    vio = [1- x.iloc[i] for i in range(n_vio)]
    norm_base = sum(vio)
    return vio[0]/norm_base * (1- x.iloc[n_vio]) + vio[1]/norm_base * x.iloc[n_vio] + vio[2]/norm_base * (1- x.iloc[n_vio+1]) + vio[3]/norm_base * x.iloc[n_vio+1]

def assign_pred_mcc_weight_group(x, n_vio=4): # assign prediction based on minimal violation
    weight_g1 = 1- min(x.iloc[2], x.iloc[3])
    weight_g0 = 1- min(x.iloc[0], x.iloc[1])
    norm_base = weight_g1 + weight_g0
    return weight_g0/norm_base * x.iloc[n_vio] + weight_g1/norm_base * x.iloc[n_vio+1]

def assign_pred_sensi(x): # assign prediction based on the group membership of sensitive attribute
    if x.iloc[0]: # sensi_col == 1, majority group
        return x.iloc[2]
    else:
        return x.iloc[1]

def eval_predicitons(data_name, seed, method, weight_base='one', res_path='../intermediate/models/',
                                 set_suffix = 'S_1', y_col='Y', sensi_col='A'):
    cur_dir = res_path + data_name + '/'

    if method == 'mcc':
        test_df = pd.read_csv(cur_dir + '-'.join(['test_vio', str(seed)]) + '.csv')

        vio_cols = ['vio_G{}_L{}'.format(group_i, label_i) for group_i in range(2) for label_i in range(2)]
        if set_suffix == 'S_1':
            features = [x for x in test_df.columns if x != y_col]
        else:
            features = [x for x in test_df.columns if x != y_col and x != sensi_col]

        test_data = test_df[features]
        opt_thres = read_json('{}thres-{}.json'.format(cur_dir, seed))['thres']

        for group_suffix in [None, 'G0', 'G1']:
            if group_suffix is None:
                model = load('{}model-{}-{}.joblib'.format(cur_dir, seed, set_suffix))
                cur_pred_col = 'Y_pred'
            else:
                model = load('{}model-{}-{}-{}.joblib'.format(cur_dir, seed, group_suffix, set_suffix))
                cur_pred_col = 'Y_pred_' + group_suffix

            test_df[cur_pred_col] = generate_model_predictions(model, test_data)


        test_df['Y_pred_min'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_min(x), axis=1)
        test_df['Y_pred_w1'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight(x), axis=1)
        test_df['Y_pred_w2'] = test_df[vio_cols + ['Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_mcc_weight_group(x), axis=1)
        test_df['Y_pred_A'] = test_df[[sensi_col, 'Y_pred_G0', 'Y_pred_G1']].apply(lambda x: assign_pred_sensi(x), axis=1)

        test_df['Y_pred_min'] = test_df['Y_pred_min'].apply(lambda x: int(x > opt_thres))
        test_df['Y_pred_w1'] = test_df['Y_pred_w1'].apply(lambda x: int(x > opt_thres))
        test_df['Y_pred_w2'] = test_df['Y_pred_w2'].apply(lambda x: int(x > opt_thres))
        test_df['Y_pred_A'] = test_df['Y_pred_A'].apply(lambda x: int(x > opt_thres))
        test_df['Y_pred'] = test_df['Y_pred'].apply(lambda x: int(x > opt_thres))

        eval_res = {}
        for pred_y, setting in zip(['Y_pred_min', 'Y_pred_w1', 'Y_pred_w2', 'Y_pred_A', 'Y_pred'], ['MCC-MIN', 'MCC-W1', 'MCC-W2', 'SEP', 'ORIG']):
            eval_res[setting] = eval_settings(test_df, sensi_col, pred_y)

        save_json(eval_res, '{}eval-{}-{}-{}.json'.format(cur_dir, seed, set_suffix, method))

    elif method in ['scc', 'omn', 'kam']:
        eval_res = {}
        test_df = pd.read_csv('{}test-{}-{}-{}-{}.csv'.format(cur_dir, seed, set_suffix, method, weight_base))
        eval_res[method.upper()] = eval_settings(test_df, sensi_col, 'Y_pred')

        save_json(eval_res, '{}eval-{}-{}-{}-{}.json'.format(cur_dir, seed, set_suffix, method, weight_base))
    else:
        raise ValueError('Input "method" is not supported. CHOOSE FROM [mcc, scc, omn, kam].')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval predictions on real data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--method", type=str,
                        help="which method to evaluate. CHOOSE FROM '[mcc, scc, omn, kam]'.")
    parser.add_argument("--base", type=str, default='one',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=9,
                        help="number of datasets over which the script is running. Default is 8 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    # datasets = ['cardio', 'bank', 'meps16', 'lsac', 'credit', 'ACSE', 'ACSP', 'ACSH', 'ACSM', 'ACSI']
    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    # datasets = ['cardio', 'bank', 'meps16', 'lsac'] # todo: run retrain for 'ACSM', 'ACSI'
    seeds = [1, 12345, 6, 2211, 15]
    datasets = ['cardio', 'bank', 'meps16', 'lsac', 'credit', 'ACSE', 'ACSP', 'ACSM', 'ACSI'] # for MCC
    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
    #          0xdeadbeef, 0xbeefcafe]


    # datasets = ['lsac']
    # seeds = [1]

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

    res_path = '../intermediate/models/'
    # for data_name in datasets:
    #     for seed in seeds:
    #         eval_predicitons(data_name, seed, args.method, args.base, res_path)

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds:
                tasks.append([data_name, seed, args.method, args.base, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(eval_predicitons, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')