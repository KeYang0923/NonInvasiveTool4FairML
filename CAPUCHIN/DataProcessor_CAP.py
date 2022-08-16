# Process real datasets using CAPUCHINE
# REQUIRE the activation of virtual enviroment that installs AIF 360. See details at https://github.com/Trusted-AI/AIF360.
import warnings
import os

from Core.indep_repair import Repair
from Modules.InformationTheory.info_theo import Info
import pandas as pd
import numpy as np

warnings.filterwarnings(action='ignore')

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


def generate_CAP_repaired_data(data_name, seed, num_atts, repair_method='MF', data_path='../data/processed/',
                               output_path='../intermediate/cap_res/'):
    if data_name in ['lawgpa', 'credit', 'UFRGS']: # for these two datasets, use the categorized version because that they originally consist of numerical attributes
        input_df = pd.read_csv(data_path + data_name + '_dense_cat.csv')
    else:
        input_df = pd.read_csv(data_path + data_name + '_dense.csv')

    train_df, _, test_df = split(input_df, seed)

    output_path = output_path + data_name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    train_df.to_csv(output_path + 'train_' + str(seed) + '.csv', index=False)
    test_df.to_csv(output_path + 'test_cat' + str(seed) + '.csv', index=False)

    data = train_df.drop(columns=num_atts)  # skip the numerical attributes
    if data_name == 'adult':
        D_features = ['sex']
        Y_features = ['Income Binary']

    elif data_name == 'german':
        D_features = ['age']
        Y_features = ['credit']

    elif data_name == 'compas':
        D_features = ['race']
        Y_features = ['two_year_recid']

    elif data_name in ['cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']:
        D_features = ['C0']
        Y_features = ['Y']
    else:
        raise ValueError(
            'The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS]')

    X_features = [x for x in data.columns if x not in D_features and x not in Y_features]
    indep = [D_features, Y_features, X_features]
    inf = Info(data)
    X = indep[0]
    Y = indep[1]
    Z = indep[2]
    mi = inf.CMI(X, Y, Z)

    rep = Repair()
    if repair_method == 'sat':
        rep.from_file_indep_repair(output_path + 'train_', X, Y, Z, method=repair_method, n_parti=100,
                                   k=seed, sample_frac=1, insert_ratio='hard', conf_weight=1)
        rep.from_file_indep_repair(output_path + 'train_', X, Y, Z, method=repair_method, n_parti=100,
                                   k=seed, sample_frac=1, insert_ratio='soft', conf_weight=1)
    else:
        rep.from_file_indep_repair(output_path + 'train_', X, Y, Z, method=repair_method, n_parti=100,
                                   k=seed, sample_frac=1, insert_ratio=1, conf_weight=2000)


if __name__ == '__main__':

    processed_file_path = '../data/processed/'
    CAP_processed_file_path = '../intermediate/cap_res/'

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]


    num_atts_mapping = {'adult': ['hours-per-week', 'age'],
                        'german': ['month', 'credit_amount'],
                        'compas': ['priors_count', 'length_of_stay'],
                        'cardio': ['X' + str(i) for i in range(1, 4)],
                        'bank': ['X' + str(i) for i in range(1, 5)],
                        'meps16': ['X' + str(i) for i in range(1, 5)],
                        'lawgpa': ['X' + str(i) for i in range(1, 3)],
                        'credit': ['X' + str(i) for i in range(1, 6)],
                        'UFRGS': ['X' + str(i) for i in range(1, 10)]
                        }

    for data_name in datasets[-1:]:
        num_atts = num_atts_mapping[data_name]
        if data_name in ['lawgpa', 'credit', 'UFRGS']: # categorize the attributes in these two datasets
            df = pd.read_csv(processed_file_path + data_name + '_dense.csv')
            num_atts = num_atts_mapping[data_name]
            k = len(num_atts)
            n_bins = 5
            labels = [i for i in range(1, n_bins + 1)]
            for i in range(1, k + 1):
                col = 'X' + str(i)
                df[col + '_cat'] = pd.cut(df[col], bins=n_bins, labels=labels)
            df.to_csv(processed_file_path + data_name + '_dense_cat.csv', index=False)

        for seed in seeds[1:]:
            generate_CAP_repaired_data(data_name, seed, num_atts, data_path=processed_file_path, output_path=CAP_processed_file_path)
            print('Using CAPUCHIN repaired ', data_name, seed)
