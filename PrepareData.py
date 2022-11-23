# Prepare all the real datasets: remove null, min-max numerical attributes, and one-hot-encoding the categorical attributes
# REQUIRE the activation of virtual enviroment that installs libraries listed in https://github.com/KeYang0923/NonInvasiveTool4FairML/blob/main/requirements.txt

import warnings
import argparse
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
import pandas as pd
import numpy as np
import json, os, pathlib

warnings.filterwarnings(action='ignore')

def read_json(file_name_with_path):
    if os.path.isfile(file_name_with_path):
        with open(file_name_with_path) as f:
            res = json.load(f)
        return res
    else:
        raise ValueError('Not exist', file_name_with_path)
def save_json(input_dict, file_path_with_name, verbose=False):
    with open(file_path_with_name, 'w') as json_file:
        json.dump(input_dict, json_file, indent=2)
    if verbose:
        print('--> Dict is saved to ', file_path_with_name + '\n')

def make_folder(file_path):
    if not os.path.exists(file_path):
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

def split(df, seed, sizes=[0.7, 0.5]):
    np.random.seed(seed)
    n = df.shape[0]
    split_point = int(sizes[0] * n)
    order = list(np.random.permutation(n))
    train_df = df.iloc[order[:split_point], :]

    vt_df = df.iloc[order[split_point:], :]
    second_n = vt_df.shape[0]
    second_order = list(np.random.permutation(second_n))
    second_split_point = int(sizes[1] * second_n)

    val_df = vt_df.iloc[second_order[:second_split_point], :]
    test_df = vt_df.iloc[second_order[second_split_point:], :]
    return train_df, val_df, test_df

def prepare_data_for_ML_models(data_name, seeds, cur_path, n_bins=10, sensi_col='A', y_col='Y',
                               bin_encode_flag='ordinal', bin_strategy = 'kmeans'):
    res_path = cur_path + '/intermediate/models/'
    data_path = cur_path + '/data/processed/'
    cur_path = cur_path + '/'
    if data_name == 'meps16':
        data_obj = MEPS(path=cur_path)
    elif data_name == 'lsac':
        data_obj = LSAC(path=cur_path)
    elif data_name == 'cardio':
        data_obj = Cardio(path=cur_path)
    elif data_name == 'credit':
        data_obj = GMCredit(path=cur_path)
    elif data_name == 'bank':
        data_obj = Bank(path=cur_path)
    elif data_name == 'ACSE':
        data_obj = ACSEmploy(path=cur_path)
    elif data_name == 'ACSP':
        data_obj = ACSPublicCoverage(path=cur_path)
    elif data_name == 'ACSH':
        data_obj = ACSHealthInsurance(path=cur_path)
    elif data_name == 'ACST':
        data_obj = ACSTravelTime(path=cur_path)
    elif data_name == 'ACSM':
        data_obj = ACSMobility(path=cur_path)
    elif data_name == 'ACSI':
        data_obj = ACSIncomePovertyRatio(path=cur_path)
    else:
        raise ValueError('The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACST", "ACSP", "ACSH", "ACSM", "ACSI"].')

    orig_df = data_obj.preprocess(data_path)
    save_json(data_obj.meta_info, '{}{}.json'.format(data_path, data_name))

    cur_dir = res_path + data_name + '/'
    make_folder(cur_dir)

    n_features = data_obj.meta_info['n_features']
    n_cond_features = len(data_obj.meta_info['continuous_features'])
    num_cols = ['X{}'.format(i) for i in range(1, n_cond_features + 1)]
    cat_cols = ['X{}'.format(i) for i in range(n_cond_features + 1, n_features)]

    for seed in seeds:
        # split data into three parts in the ratio of 70%, 15%, 15%
        tr_df, vl_df, te_df = split(orig_df, seed)
        tr_df.to_csv('{}train-{}.csv'.format(cur_dir, seed), index=False) # keep index for sanity check of random splits
        vl_df.to_csv('{}val-{}.csv'.format(cur_dir, seed), index=False)
        te_df.to_csv('{}test-{}.csv'.format(cur_dir, seed), index=False)

        # bin the three parts for CAPUCHIN
        train_data = tr_df[num_cols].copy()
        val_data = vl_df[num_cols].copy()
        test_data = te_df[num_cols].copy()

        # bin the numerical attributes
        est = KBinsDiscretizer(n_bins=n_bins, encode=bin_encode_flag, strategy=bin_strategy)
        est.fit(train_data)
        cat_train_data = est.transform(train_data)
        cat_val_data = est.transform(val_data)
        cat_test_data = est.transform(test_data)

        bin_train_df = pd.DataFrame(columns=num_cols, data=cat_train_data)
        bin_val_df = pd.DataFrame(columns=num_cols, data=cat_val_data)
        bin_test_df = pd.DataFrame(columns=num_cols, data=cat_test_data)

        for col_i in cat_cols + [y_col, sensi_col]:
            bin_train_df[col_i] = list(tr_df[col_i])
            bin_val_df[col_i] = list(vl_df[col_i])
            bin_test_df[col_i] = list(te_df[col_i])

        bin_test_df.to_csv('{}train-{}-bin.csv'.format(cur_dir, seed), index=False)
        bin_val_df.to_csv('{}val-{}-bin.csv'.format(cur_dir, seed), index=False)
        bin_test_df.to_csv('{}test-{}-bin.csv'.format(cur_dir, seed), index=False)

        save_json({'n_bins': n_bins, 'encode': bin_encode_flag, 'strategy': bin_strategy},
                  '{}par-bin-{}.json'.format(cur_dir, seed))


class Dataset():
    # preprocess real datasets as below
    # drop null values if any, sample 100000 if more than this value for experiment efficiency
    # encode positive outcome as 1, protected group as 0
    # rename features, labels, and sensitive attribute to 'X#', 'Y', and 'A' correspondingly
    # produce a json file to store meta information includes size, attributes with continuous values (unique values more than 8), target column, sensitive attribute
    def __init__(self, df, name, label_col, fav_mapping, fav_value, sensi_col, sensi_value_mapping, sensi_transform):
        self.name = name
        self.numerical_cols = []
        self.categorical_cols = []
        self.label_col = label_col
        self.fav_mapping = fav_mapping
        self.fav_value = fav_value

        self.sensi_col = sensi_col
        self.sensi_value_mapping = sensi_value_mapping
        self.sensi_transform = sensi_transform
        self.df = df
        self.meta_info = {'target': label_col,
                          'target_positive_value': fav_value,
                          'sensitive_feature': sensi_col}
    def preprocess(self, output_path, num_threshold=8, sample_activate_n=500000):
        df = self.df.dropna()

        if df.shape[0] > sample_activate_n:  # sample rows for efficiency
            print('Activate sampling over ', self.name, 'for experiment efficiency')

            cur_df = df.sample(n=sample_activate_n, random_state=0).copy()
            cur_df.reset_index(inplace=True)
            cur_df.drop(columns=['index'], inplace=True)
        else:
            cur_df = df.copy()

        num_atts = []
        cat_atts = []
        # the columns having more than 'num_threshold' distinguishing values are treated as continuous.
        # features = list(set(cur_df.columns).difference([self.sensi_col, self.label_col]))
        for col_i in cur_df.columns:
            if col_i == self.sensi_col or col_i == self.label_col:
                pass
            else:
                if len(cur_df[col_i].unique()) >= num_threshold:
                    num_atts.append(col_i)
                else:
                    cat_atts.append(col_i)


        if self.fav_mapping is not None:  # binarize target column
            cur_df['Y'] = cur_df[self.label_col].apply(lambda x: int(x == self.fav_mapping))
        else:
            cur_df['Y'] = cur_df[self.label_col]
            
        if self.sensi_transform:
            cur_df['A'] = cur_df[self.sensi_col].replace(self.sensi_value_mapping)
        else:
            cur_df['A'] = cur_df[self.sensi_col]

        cur_df = cur_df.astype({'A': np.int, 'Y': np.int})
        
        protected_groups = [key for key, value in self.sensi_value_mapping.items() if value == 0]
        col_name_mapping = {col: 'X' + str(i + 1) for i, col in enumerate(num_atts + cat_atts)}
        cur_df.rename(columns=col_name_mapping, inplace=True)

        group_df = cur_df.query('A==0')
        pos_df = group_df.query('Y==1')

        self.meta_info.update({'size': cur_df.shape[0],
                                 'n_features': len(cat_atts)+len(num_atts) + 1,
                                 'continuous_features': num_atts,
                                 'categorical_features': cat_atts,
                                 'protected_group': protected_groups,
                                 'group_perc': round(group_df.shape[0]/cur_df.shape[0]*100, 1),
                                 'pos_perc': round(pos_df.shape[0]/group_df.shape[0]*100, 1),
                                 'feature_name_mapping': col_name_mapping
                             })

        output_df = cur_df[['X' + str(i) for i in range(1, len(cat_atts)+len(num_atts) + 1)] + ['Y', 'A']]
        output_df.to_csv(output_path + self.name + '.csv', index=False)
        print('--> Processed data is saved to ', output_path + self.name + '.csv\n')
        return output_df

class Cardio(Dataset):
    # this dataset is from Kaggle https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
    # And is also used in the paper "Learning to Validate the Predictions of Black Box Classifiers on Unseen Data"

    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/cardio/cardio_train.csv', sep=';')
        except IOError as err:
            print("IOError: {}".format(err))

        raw_df['bmi'] = raw_df['weight'] / (.01 * raw_df['height']) ** 2
        raw_df['age_in_years'] = raw_df['age'] / 365

        raw_df['age_b'] = raw_df['age_in_years'].apply(lambda x: int(x >= 45))
        raw_df = raw_df[['bmi', 'ap_hi', 'ap_lo', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_b', 'cardio']]
        # categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        # numerical_columns = ['bmi', 'ap_hi', 'ap_lo']

        label_col = 'cardio'
        fav_transform = None # no need to transform as the original target column is binary
        fav_meta = 'having cardio disease' #  for meta information

        sensi_col = 'age_b' # 0 for younger and 1 for older
        sensi_col_mapping = {'age < 45': 0, 'age >= 45': 1} #  for meta information
        sensi_transform_flag = False # no need to transform as the original sensitive column is binary

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping, sensi_transform_flag)
        else:
            super().__init__(raw_df, 'cardio', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping, sensi_transform_flag)

class LSAC(Dataset):
    # this dataset is from OminiFair

    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path + 'data/lsac/lsac.csv') # the data is derived from running the above code
        except IOError as err:
            print("IOError: {}".format(err))

        def group_race(x):
            # if x == "White":
            #     return 1.0
            # else:
            #     return 0.0
            if x == "Black":
                return 0.0
            elif x == "White":
                return 1.0
            else:
                return -1

        raw_df['sex'] = raw_df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        raw_df['race'] = raw_df['race'].apply(lambda x: group_race(x))
        raw_df = raw_df[raw_df['race'] >= 0]

        raw_df['pass_bar'] = raw_df['pass_bar'].replace({'Passed': 1.0, 'Failed_or_not_attempted': 0.0})
        raw_df['isPartTime'] = raw_df['isPartTime'].replace({'Yes': 1.0, 'No': 0.0})

        raw_df = raw_df[['zfygpa', 'zgpa', 'DOB_yr', 'isPartTime', 'sex', 'race', 'cluster_tier', 'family_income', 'lsat', 'ugpa', 'weighted_lsat_ugpa', 'pass_bar']]

        label_col = 'pass_bar'
        fav_transform = None  # no need to transform as the original target column is binary
        fav_meta = 'Passed'  # for meta information

        sensi_col = 'race'
        sensi_col_mapping = {'White': 1, 'Black': 0}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping, sensi_transform_flag)
        else:
            super().__init__(raw_df, 'lsac', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping, sensi_transform_flag)

class GMCredit(Dataset):
    # this dataset is from Kaggle competition and for classification task of predicting the probability that somebody will experience financial distress in the next two years.
    # We use the training variants in our experiments. See details at https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/credit/GiveMeSomeCredit_training.csv')

        except IOError as err:
            print("IOError: {}".format(err))

        raw_df = raw_df[['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'SeriousDlqin2yrs', 'age']]
        raw_df['age'] = raw_df['age'].apply(lambda x: int(x >= 35))

        label_col = 'SeriousDlqin2yrs' # presence of financial stress
        fav_transform = None  # no need to transform as the original target column is binary
        fav_meta = 'serious delay in 2 years'  # for meta information

        sensi_col = 'age'
        sensi_col_mapping = {'age >= 35': 1, 'age < 35': 0}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping, sensi_transform_flag)
        else:
            super().__init__(raw_df, 'credit', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping, sensi_transform_flag)

class MEPS(Dataset):
    # this dataset is from AIF360 and for classification and regression tasks, we binarize labels as in AIF 360: https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/meps_dataset_panel21_fy2016.py
    # REQUIRE RUNNING R SCRIPT FIRST TO EXTRACT THE CSV, see details at https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/meps/README.md
    def default_preprocessing(self, df):
        """
        Preprocess steos from AIF 360
        1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
          and 'Non-White' otherwise
        2. Restrict to Panel 21
        3. RENAME all columns that are PANEL/ROUND SPECIFIC
        4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
        5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
        """

        def race(row):
            if ((row['HISPANX'] == 2) and (
                    row['RACEV2X'] == 1)):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
                return 'White'
            return 'Non-White'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df = df.rename(columns={'RACEV2X': 'RACE'})

        df = df[df['PANEL'] == 21]

        # RENAME COLUMNS
        df = df.rename(columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                                'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                                'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                                'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                                'POVCAT16': 'POVCAT', 'INSCOV16': 'INSCOV'})

        df = df[df['REGION'] >= 0]  # remove values -1
        df = df[df['AGE'] >= 0]  # remove values -1

        df = df[df['MARRY'] >= 0]  # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                     'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                     'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                     'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                     'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(
            1)]  # for all other categorical features, remove values < -1

        def utilization(row):
            return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']

        df['TOTEXP16'] = df.apply(lambda row: utilization(row), axis=1)

        lessE = df['TOTEXP16'] < 10.0
        df.loc[lessE, 'TOTEXP16'] = 0.0
        moreE = df['TOTEXP16'] >= 10.0
        df.loc[moreE, 'TOTEXP16'] = 1.0

        df = df.rename(columns={'TOTEXP16': 'UTILIZATION'})
        return df

    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/meps16/h192.csv', sep=',')
        except IOError as err:
            print("IOError: {}".format(err))

        raw_df = self.default_preprocessing(raw_df)

        columns = ['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                   'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                   'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                   'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                   'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                   'PCS42',
                   'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION', 'PERWT16F']

        raw_df = raw_df[columns]

        categorical_columns = ['REGION', 'SEX', 'MARRY',
                               'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                               'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                               'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                               'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PHQ242',
                               'EMPST', 'POVCAT', 'INSCOV']
        numerical_columns = ['PERWT16F', 'MCS42', 'PCS42', 'K6SUM42']
        raw_df = raw_df[categorical_columns+numerical_columns+['UTILIZATION', 'RACE']]

        label_col = 'UTILIZATION'
        fav_transform = None  # no need to transform as the original target column is binary
        fav_meta = 'UTILIZATION >= 10'  # for meta information

        sensi_col = 'RACE'
        sensi_col_mapping = {'White': 1, 'Non-White': 0}
        sensi_transform_flag = True

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'meps16', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)


class Bank(Dataset):
    # this dataset is from AIF 360 and for classification tasks. See details at https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/bank_dataset.py
    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/bankmarketing/bank-additional-full.csv', sep=';', na_values='unknown')
        except IOError as err:
            print("IOError: {}".format(err))

        raw_df = raw_df.dropna()
        numerical_columns = ['duration', 'age']
        categorical_columns = ['housing', 'loan', 'contact', 'marital']
        extra_num_atts = ['job', 'education', 'month', 'day_of_week']
        le = LabelEncoder()
        for col in extra_num_atts+categorical_columns:
            raw_df[col] = le.fit_transform(raw_df[col])

        raw_df = raw_df.dropna()

        raw_df['age'] = raw_df['age'].apply(lambda x: int(x > 30))
        raw_df = raw_df[categorical_columns+numerical_columns+extra_num_atts+['y']]

        label_col = 'y'
        fav_transform = 'yes'
        fav_meta = 'whether a client will make a deposit subscription or not'  # for meta information


        sensi_col = 'age'
        sensi_col_mapping = {'age <= 30': 0, 'age > 30': 1}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'bank', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)


class ACSEmploy(Dataset):
    # this dataset is from folktables. See details at https://github.com/zykls/folktables
    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/ACS/ACS_Employ_race.csv')
        except IOError as err:
            print("IOError: {}".format(err))


        label_col = 'Y'
        fav_transform = None
        fav_meta = 'ESR'  # for meta information

        sensi_col = 'RACE'
        sensi_col_mapping = {'Black': 0, 'White': 1}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'ACSE', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)

class ACSPublicCoverage(Dataset):
    # this dataset is from folktables. See details at https://github.com/zykls/folktables
    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/ACS/ACS_PublicCoverage_race.csv')
        except IOError as err:
            print("IOError: {}".format(err))


        label_col = 'Y'
        fav_transform = 0
        fav_meta = 'Private Insurance'  # for meta information

        sensi_col = 'RACE'
        sensi_col_mapping = {'Black': 0, 'White': 1}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'ACSP', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
class ACSHealthInsurance(Dataset):
    # this dataset is from folktables. See details at https://github.com/zykls/folktables
    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/ACS/ACS_HealthInsurance_race.csv')
        except IOError as err:
            print("IOError: {}".format(err))


        label_col = 'Y'
        fav_transform = None
        fav_meta = 'HINS2'  # for meta information

        sensi_col = 'RACE'
        sensi_col_mapping = {'Black': 0, 'White': 1}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'ACSH', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)

class ACSTravelTime(Dataset):
    # this dataset is from folktables. See details at https://github.com/zykls/folktables
    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/ACS/ACS_TravelTime_race.csv')
        except IOError as err:
            print("IOError: {}".format(err))


        label_col = 'Y'
        fav_transform = 0
        fav_meta = 'JWMNP < 20'  # for meta information

        sensi_col = 'RACE'
        sensi_col_mapping = {'Black': 0, 'White': 1}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'ACST', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
class ACSMobility(Dataset):
    # this dataset is from folktables. See details at https://github.com/zykls/folktables
    # this class produces the version that includes the features for ML models
    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/ACS/ACS_Mobility_race.csv')
        except IOError as err:
            print("IOError: {}".format(err))


        label_col = 'Y'
        fav_transform = 0 # reverse the label
        fav_meta = 'not MIG'  # for meta information

        sensi_col = 'RACE'
        sensi_col_mapping = {'Black': 0, 'White': 1}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'ACSM', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)

class ACSIncomePovertyRatio(Dataset):
    # this dataset is from folktables. See details at https://github.com/zykls/folktables
    # this class produces the version that includes the features for ML models
    def __init__(self, name=None, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/ACS/ACS_IncomePovertyRatio_race.csv')
        except IOError as err:
            print("IOError: {}".format(err))


        label_col = 'Y'
        fav_transform = None
        fav_meta = 'POVPIP < 250'  # for meta information

        sensi_col = 'RACE'
        sensi_col_mapping = {'Black': 0, 'White': 1}
        sensi_transform_flag = False

        if name is not None:
            super().__init__(raw_df, name, label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)
        else:
            super().__init__(raw_df, 'ACSI', label_col, fav_transform, fav_meta, sensi_col, sensi_col_mapping,
                             sensi_transform_flag)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Prepare data for ML models")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system execution or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--bin_n", type=int, default=10,
                        help="number of binns in categorizing data. Required for CAPUCHIN. Default is 10 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac', 'bank', 'cardio', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

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

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            tasks.append([data_name, seeds, repo_dir, args.bin_n])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(prepare_data_for_ML_models, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')
