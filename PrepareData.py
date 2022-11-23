# Prepare all the real datasets: remove null, min-max numerical attributes, and one-hot-encoding the categorical attributes
# REQUIRE the activation of virtual enviroment that installs libraries listed in https://github.com/KeYang0923/NonInvasiveTool4FairML/blob/main/requirements.txt

import warnings
from sklearn import preprocessing
# import argparse

import pandas as pd
import numpy as np
import json, os
warnings.filterwarnings(action='ignore')

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
        features = list(set(cur_df.columns).difference([self.sensi_col, self.label_col]))
        for col_i in features:
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
                                 'n_features': len(features) + 1,
                                 'continuous_features': num_atts,
                                 'categorical_features': cat_atts,
                                 'protected_group': protected_groups,
                                 'group_perc': round(group_df.shape[0]/cur_df.shape[0]*100, 1),
                                 'pos_perc': round(pos_df.shape[0]/group_df.shape[0]*100, 1),
                                 'feature_name_mapping': col_name_mapping
                             })

        output_df = cur_df[['X' + str(i) for i in range(1, len(features) + 1)] + ['Y', 'A']]
        output_df.to_csv(output_path + self.name + '.csv', index=False)
        print('--> Processed data is saved to ', output_path + self.name + '.csv\n')
    def get_count(self, order_cols, place_holder='X1'):
        print(self.df[order_cols + [place_holder]].groupby(by=order_cols).count())

    def save_json(self, file_path, verbose=True):
        with open(file_path + self.name + '.json', 'w') as json_file:
            json.dump(self.meta_info, json_file, indent=2)
        if verbose:
            print('--> Meta information is saved to ', file_path + self.name + '.json\n')

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
        le = preprocessing.LabelEncoder()
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
    # initiate objects for real datasets and proprocess it with meta information extracted
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = repo_dir + '/data/processed/'
    # cardio = Cardio(path=repo_dir+'/')
    # cardio.preprocess(save_path)
    # cardio.save_json(save_path)
    #
    # lsac = LSAC(path=repo_dir+'/')
    # lsac.preprocess(save_path)
    # lsac.save_json(save_path)
    #
    # gmcredit = GMCredit(path=repo_dir+'/')
    # gmcredit.preprocess(save_path)
    # gmcredit.save_json(save_path)

    # meps = MEPS(path=repo_dir+'/')
    # meps.preprocess(save_path)
    # meps.save_json(save_path)

    # bank = Bank(path=repo_dir+'/')
    # bank.preprocess(save_path)
    # bank.save_json(save_path)

    # acse = ACSEmploy(path=repo_dir+'/')
    # acse.preprocess(save_path)
    # acse.save_json(save_path)

    # acsp = ACSPublicCoverage(path=repo_dir+'/')
    # acsp.preprocess(save_path)
    # acsp.save_json(save_path)

    # acsh = ACSHealthInsurance(path=repo_dir+'/')
    # acsh.preprocess(save_path)
    # acsh.save_json(save_path)

    # acst = ACSTravelTime(path=repo_dir+'/')
    # acst.preprocess(save_path)
    # acst.save_json(save_path)

    # acsm = ACSMobility(path=repo_dir+'/')
    # acsm.preprocess(save_path)
    # acsm.save_json(save_path)

    # acsi = ACSIncomePovertyRatio(path=repo_dir+'/')
    # acsi.preprocess(save_path)
    # acsi.save_json(save_path)