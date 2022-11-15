import numpy as np
import os, pathlib, json
from joblib import load

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error


def eval_predictions(y_true, y_pred, label_order=[0, 1], metrics=['AUC', 'ACC', 'ERR', 'FPR', 'FNR', 'PR', 'TPR', 'TNR', 'TP', 'FN', 'TN', 'FP', 'SR']):
    if metrics == 'MAE':
        return mean_absolute_error(y_true, y_pred)
    else:
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


def load_model(dir, seed, sensi_suffix, group_suffix=None):
    if group_suffix is not None:
        model = load(dir + '-'.join(['model', str(seed), group_suffix, sensi_suffix]) + '.joblib')
    else:
        model = load(dir + '-'.join(['model', str(seed), sensi_suffix]) + '.joblib')

    return model

def split(data, seed, sizes=[0.7, 0.5]):
    np.random.seed(seed)
    n = data.shape[0]
    split_point = int(sizes[0] * n)
    order = list(np.random.permutation(n))
    train_data = data.iloc[order[:split_point], :]

    vt_data = data.iloc[order[split_point:], :]
    second_n = vt_data.shape[0]
    second_order = list(np.random.permutation(second_n))
    second_split_point = int(sizes[1] * second_n)

    val_data = vt_data.iloc[second_order[:second_split_point], :]
    test_data = vt_data.iloc[second_order[second_split_point:], :]
    return train_data, val_data, test_data

def format_print(str_input, output_f=None):
    if output_f is not None:
        print(str_input, file=output_f)
    else:
        print(str_input)

def format_filter_df(df, filter_cols, keep_values, default_opt='and'):
    if default_opt == 'or':
        query_base = ' or '
    else:
        query_base = ' and '
    if len(filter_cols) == len(keep_values):
        if len(filter_cols) > 1:
            query_str = query_base.join([x + '=="' + y + '"' for x, y in zip(filter_cols, keep_values)])
        elif len(filter_cols) == 1:
            query_str = '{}=="{}"'.format(filter_cols[0], keep_values[0])
        else:
            raise ValueError('No columns and values specified!')
    else:
        if len(keep_values) == 0 or len(filter_cols) == 0:
            raise ValueError('No values specified!')
        elif len(keep_values) == 1:
            query_str = query_base.join([x + '=="' + keep_values[0] + '"' for x in filter_cols])
        elif len(filter_cols) == 1:
            query_str = ' or '.join([filter_cols[0] + '=="' + y + '"' for y in keep_values])

    return df.query(query_str)

def make_folder(file_path):
    if not os.path.exists(file_path):
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        # print('Make folder ', file_path)

def save_to_disc(file_name_with_path, df, index_flag=False, excluded_cols=[]):
    if not os.path.exists(file_name_with_path):
        directory = os.path.dirname(file_name_with_path)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    if excluded_cols:
        df.drop(columns=excluded_cols).to_csv(file_name_with_path, index=index_flag)
    else:
        df.to_csv(file_name_with_path, index=index_flag)
    print('--> Data is saved to ', file_name_with_path, '\n')

def save_json(results, file_name_with_path, verbose=False):
    with open(file_name_with_path + '.json', 'w') as json_file:
        json.dump(results, json_file)
    if verbose:
        print('--> Json is saved to ', file_name_with_path + '.json\n\n')

def read_json(file_name_with_path):
    if os.path.isfile(file_name_with_path + '.json'):
        with open(file_name_with_path + '.json') as f:
            res = json.load(f)
        return res
    else:
        return None
