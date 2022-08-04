# Prepare real datasets: Cardio diseases, Bank marketing, MEPS16, Credit, Law School GPA
# REQUIRE the activation of virtual enviroment that installs libraries listed in https://github.com/KeYang0923/NonInvasiveTool4FairML/blob/main/requirements.txt

import warnings
from sklearn import preprocessing
import argparse

import pandas as pd
import numpy as np
warnings.filterwarnings(action='ignore')

class Dataset():
    # preprocess real datasets that are not used in the IBM AIFairness 360
    # one-hot encode categorical features, normalize the numerical features, and drop null values if any
    # rename features, labels, and sensitive attribute to 'X#', 'Y', and 'C0' correspondingly
    def __init__(self, df, name, num_atts, label_col, posi_label, cluster_col, cluster_mapping,
                 extra_num_atts=None, cat_atts=None):
        self.name = name
        self.numerical_columns = num_atts
        self.label_col = label_col
        self.posi_label = posi_label
        self.cluster_col = cluster_col
        self.cluster_mapping = cluster_mapping
        self.df = df.copy()

        self.extra_num_atts = extra_num_atts  # for training ML models
        self.cat_atts = cat_atts

    def preprocess(self, n_clusters=2, dummy_flag=True, sample_cluster=10000, random_state=0, sample_activate_n=100000):

        if self.df.shape[0] > sample_activate_n:  # sample rows for each cluster for efficiency
            print('Activate sampling at ', self.name)
            if self.cluster_mapping:
                values = self.cluster_mapping.keys()
            else:
                values = range(n_clusters)

            cur_df = pd.DataFrame()
            for vi in values:
                vi_df = self.df[self.df[self.cluster_col] == vi]
                if vi_df.shape[0] > sample_cluster:
                    cur_df = pd.concat([cur_df, vi_df.sample(n=sample_cluster, random_state=random_state)])
                else:
                    cur_df = pd.concat([cur_df, vi_df])
            cur_df.reset_index()
        else:
            cur_df = self.df.copy()

        cur_df = cur_df.dropna()

        cur_df[self.numerical_columns] = (cur_df[self.numerical_columns] - cur_df[self.numerical_columns].mean()) / \
                                         cur_df[self.numerical_columns].std()

        if self.posi_label is not None and self.label_col is not None:  # for classifications
            cur_df['Y'] = cur_df[self.label_col].apply(lambda x: int(x == self.posi_label))
        elif self.label_col is not None:
            cur_df['Y'] = cur_df[self.label_col]
        else:
            pass
        if self.cluster_mapping is not None:
            cur_df['C0'] = cur_df[self.cluster_col].replace(self.cluster_mapping)
        else:
            cur_df['C0'] = cur_df[self.cluster_col]

        cur_df = cur_df.astype({'C0': np.int})

        cols_names = []
        cols_names += self.numerical_columns

        if self.extra_num_atts:
            cur_df[self.extra_num_atts] = (cur_df[self.extra_num_atts] - cur_df[self.extra_num_atts].mean()) / cur_df[
                self.extra_num_atts].std()
            cols_names += self.extra_num_atts
            cur_df = cur_df.fillna(-1)

        if self.cat_atts:
            for col_i in self.cat_atts:
                cur_df[col_i] = cur_df[col_i].astype('category')
            if dummy_flag:
                encoded_cats_df = pd.get_dummies(cur_df[self.cat_atts])

                cat_cols = list(encoded_cats_df.columns)
                #             print(cat_cols)
                cur_df = cur_df[cols_names + ['Y', 'C0']]
                cur_df = pd.concat([cur_df, encoded_cats_df], axis=1)
                cols_names += cat_cols
                cur_df = cur_df.fillna(-1)
            else:
                cols_names += self.cat_atts
                cur_df = cur_df.dropna()


        cur_df.rename(columns={col: 'X' + str(i + 1) for i, col in enumerate(cols_names)}, inplace=True)

        return cur_df[['X' + str(i) for i in range(1, len(cols_names) + 1)] + ['Y', 'C0']]

    def get_count(self, df, order_cols, place_holder='X1'):
        print(df[order_cols + [place_holder]].groupby(by=order_cols).count())


class LawGPA(Dataset):
    # this dataset is from AIF360 and for regression task. We use it for classification tasks by binarizing GPA.
    # https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/law_school_gpa_dataset.py

    def __init__(self, name=None):
        try:
            # import tempeh.configurations as tc
            # dataset = tc.datasets["lawschool_gpa"]()
            # X_train, X_test = dataset.get_X(format=pd.DataFrame)
            # y_train, y_test = dataset.get_y(format=pd.Series)
            # A_train, A_test = dataset.get_sensitive_features(name='race',
            #                                                  format=pd.Series)
            # all_train = pd.concat([X_train, y_train, A_train], axis=1)
            # all_test = pd.concat([X_test, y_test, A_test], axis=1)
            #
            # raw_df = pd.concat([all_train, all_test], axis=0)
            raw_df = pd.read_csv('../data/lawgpa/data.csv') # the data is derived from running the above code
        except IOError as err:
            print("IOError: {}".format(err))

        categorical_columns = None
        numerical_columns = ['lsat', 'ugpa']

        label_col = 'zfygpa'
        extra_num_atts = None

        posi_label = None

        cluster_col = 'race'
        cluster_col_mapping = {'white': 1, 'black': 0}
        if name is not None:
            super().__init__(raw_df, name, numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping)
        else:
            super().__init__(raw_df, 'lawgpa', numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping)

class UFRGS(Dataset):
    # this dataset is from Harvard Dataverse. See details at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O35FW8
    # It consists of entrance exam scores of students applying to a university in Brazil (Federal University of Rio Grande do Sul), along with the students' GPAs during the first three semesters at university.

    def __init__(self, name=None):
        try:
            raw_df = pd.read_csv('../data/UFRGS/data.csv')

        except IOError as err:
            print("IOError: {}".format(err))

        categorical_columns = None
        numerical_columns = ['physics', 'biology', 'history', 'SecondLanguage', 'geography', 'literature',
                             'PortugueseEssay', 'math', 'chemistry']

        raw_df['GPA'] = raw_df['GPA'].apply(lambda x: int(x >= 3.0))
        label_col = 'GPA' # whether GPA in the first three semesters is greater than 3.0

        posi_label = None

        cluster_col = 'gender'  # 0 denotes female and 1 denotes male.
        cluster_col_mapping = None
        if name is not None:
            super().__init__(raw_df, name, numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping)
        else:
            super().__init__(raw_df, 'UFRGS', numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping)


class GMCredit(Dataset):
    # this dataset is from Kaggle competition and for classification task of predicting the probability that somebody will experience financial distress in the next two years.
    # We use the training variants in our experiments. See details at https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

    def __init__(self, name=None):
        try:
            raw_df = pd.read_csv('../data/credit/GiveMeSomeCredit_training.csv')

        except IOError as err:
            print("IOError: {}".format(err))

        categorical_columns = None
        numerical_columns = ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse',
                             'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate']

        raw_df['age'] = raw_df['age'].apply(lambda x: int(x < 65))
        label_col = 'SeriousDlqin2yrs' # whether a loan is granted

        posi_label = None

        cluster_col = 'age'
        cluster_col_mapping = None
        if name is not None:
            super().__init__(raw_df, name, numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping)
        else:
            super().__init__(raw_df, 'credit', numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping)


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

        categorical_columns = ['REGION', 'SEX', 'MARRY',
                               'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                               'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                               'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                               'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PHQ242',
                               'EMPST', 'POVCAT', 'INSCOV']
        columns = ['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                   'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                   'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                   'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                   'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                   'PCS42',
                   'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION', 'PERWT16F']

        raw_df = raw_df[columns]
        numerical_columns = ['PERWT16F', 'MCS42']

        label_col = 'UTILIZATION'
        extra_num_atts = ['PCS42', 'K6SUM42']

        posi_label = None

        cluster_col = 'RACE'
        cluster_col_mapping = {'White': 1, 'Non-White': 0}
        if name is not None:
            super().__init__(raw_df, name, numerical_columns, label_col, posi_label, cluster_col,
                             cluster_col_mapping,
                             extra_num_atts=extra_num_atts, cat_atts=categorical_columns)
        else:
            super().__init__(raw_df, 'meps16', numerical_columns, label_col, posi_label, cluster_col,
                             cluster_col_mapping,
                             extra_num_atts=extra_num_atts, cat_atts=categorical_columns)

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

        raw_df['age_b'] = raw_df['age_in_years'].apply(lambda x: int(x >= 55))

        categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        numerical_columns = ['bmi', 'ap_hi', 'ap_lo']

        label_col = 'cardio'
        posi_label = 1 # the presence of a heart disease

        cluster_col = 'age_b' # 0 for younger and 1 for older
        cluster_col_mapping = None

        if name is not None:
            super().__init__(raw_df, name, numerical_columns, label_col, posi_label, cluster_col,
                             cluster_col_mapping, cat_atts=categorical_columns)
        else:
            super().__init__(raw_df, 'cardio', numerical_columns, label_col, posi_label, cluster_col,
                             cluster_col_mapping, cat_atts=categorical_columns)


class BankDense(Dataset):
    # this dataset is from AIF 360 and for classification tasks. See details at https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/bank_dataset.py
    # this class produces the version that includes the numerical attributes
    def __init__(self, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/bankmarketing/bank-additional-full.csv', sep=';', na_values='unknown')
        except IOError as err:
            print("IOError: {}".format(err))

        raw_df = raw_df.dropna()
        categorical_columns = ['housing', 'loan', 'contact', 'marital']
        extra_num_atts = ['job', 'education', 'month', 'day_of_week']
        le = preprocessing.LabelEncoder()
        for col in extra_num_atts:
            raw_df[col] = le.fit_transform(raw_df[col])

        raw_df = raw_df.dropna()
        numerical_columns = ['duration', 'age']

        label_col = 'y'
        posi_label = 'yes'
        cluster_col = 'marital'
        cluster_col_mapping = {'married': 0, 'single': 1, 'divorced': 0}

        super().__init__(raw_df, 'bank_dense', numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping,
                         extra_num_atts=extra_num_atts, cat_atts=categorical_columns)

class BankFeatures(Dataset):
    # this dataset is from AIF 360 and for classification tasks. See details at https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/bank_dataset.py
    # this class produces the version that includes the features for ML models
    def __init__(self, path='../'):
        try:
            raw_df = pd.read_csv(path+'data/bankmarketing/bank-additional-full.csv', sep=';', na_values='unknown')
        except IOError as err:
            print("IOError: {}".format(err))

        raw_df = raw_df.dropna()

        categorical_columns = ['housing', 'loan', 'contact', 'marital', 'job', 'education', 'month', 'day_of_week']
        numerical_columns = ['duration', 'age']

        label_col = 'y'
        posi_label = 'yes'
        cluster_col = 'marital'
        cluster_col_mapping = {'married': 0, 'single': 1, 'divorced': 0}

        super().__init__(raw_df, 'bank_features', numerical_columns, label_col, posi_label, cluster_col, cluster_col_mapping,
                         cat_atts=categorical_columns)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate models over erroneous test data")
    parser.add_argument("--data", type=str,
                        help="dataset to simulate all the error rates. Use 'all' for all the datasets. OR choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit] for different datasets.")

    args = parser.parse_args()

    datasets = ['cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']


    if args.data is None:
        raise ValueError('The input "data" is requried. Use "all" for all the datasets. OR choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS] for different datasets.')

    elif args.data == 'all':
        print('To avoid the size limit of GitHub, we do not keep the raw data for MEPS. First retreive the raw data by running the R script as in "../data/meps16/generate_data.R".')

        datasets_objs = [Cardio('cardio_dense'), Cardio('cardio_features'),
                        BankDense(), BankFeatures(),
                        MEPS('meps16_dense'), MEPS('meps16_features'),
                        LawGPA('lawgpa_dense'), LawGPA('lawgpa_features'),
                        GMCredit('credit_dense'), GMCredit('credit_features'),
                        UFRGS('UFRGS_dense'), UFRGS('UFRGS_features')]
    else:
        if args.data not in datasets:
            raise ValueError(
                'The input "data" is requried. Use "all" for all the datasets. OR choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS] for different datasets.')
        else:
            if args.data == 'bank':
                datasets_objs = [BankDense(), BankFeatures()]
            elif args.data == 'meps16':
                raise ValueError(
                    'To avoid the size limit of GitHub, we do not keep the raw data for MEPS. DO retreive the raw data by running the R script as in "../data/meps16/generate_data.R". Then, uncomment the code in line 407 of this script to run the preprocessing for MEPS dataset.')
            else: # Uncomment the row for "meps16" when this script is used for MEPS dataset.
                datasets_objs_mapping = {'cardio': [Cardio('cardio_dense'), Cardio('cardio_features')],
                                         # 'meps16': [MEPS('meps16_dense'), MEPS('meps16_features')],
                                         'lawgpa': [LawGPA('lawgpa_dense'), LawGPA('lawgpa_features')],
                                         'credit': [GMCredit('credit_dense'), GMCredit('credit_features')],
                                         'UFRGS': [UFRGS('UFRGS_dense'), UFRGS('UFRGS_features')]}
                datasets_objs = datasets_objs_mapping[args.data]

    processed_file_path = '../data/processed/'

    for dataset in datasets_objs:
        df = dataset.preprocess()
        df.to_csv(processed_file_path + dataset.name + '.csv', index=False)
