# Get benchmark datasets Adult income, German credit, and COMPAS from IBM AIFairness 360
# REQUIRE the pre-installment of AIF 360. See details at https://github.com/Trusted-AI/AIF360.
# REQUIRE the activation of virtual environment that installed AIF 360 to run this python file.

import warnings
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from sklearn import preprocessing

import pandas as pd
import numpy as np
warnings.filterwarnings(action='ignore')


def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        # Group age by decade
        df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def group_edu(x):
            if x <= 5:
                return '<6'
            elif x >= 13:
                return '>12'
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
        df['Education Years'] = df['Education Years'].astype('category')

        # Limit age range
        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        # Rename income variable
        df['Income Binary'] = df['income-per-year']
        df['Income Binary'] = df['Income Binary'].replace(to_replace='>50K.', value='>50K', regex=True)
        df['Income Binary'] = df['Income Binary'].replace(to_replace='<=50K.', value='<=50K', regex=True)

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Income Binary'] == '<=50K']
            df_1 = df[df['Income Binary'] == '>50K']
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.concat([df_0, df_1])
        return df

    XD_features = ['Age (decade)', 'Education Years', 'sex', 'race', 'age', 'hours-per-week']
    D_features = ['sex', 'race'] if protected_attributes is None else protected_attributes
    Y_features = ['Income Binary']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['Age (decade)', 'Education Years']

     # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "race": {1.0: 'White', 0.0: 'Non-white'}}

    return AdultDataset(
        label_name=Y_features[0],
        favorable_classes=['>50K', '>50K.'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def load_preproc_data_compas(protected_attributes=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """

        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
                 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix,:]
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                                pd.to_datetime(df['c_jail_in'])).apply(
                                                        lambda x: x.days)

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

        # Restrict the features to use
        dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count','is_recid',
                'two_year_recid','length_of_stay']].copy()

        # Quantize priors count between 0, 1-3, and >3
        def quantizePrior(x):
            if x <=0:
                return '0'
            elif 1<=x<=3:
                return '1 to 3'
            else:
                return 'More than 3'

        # Quantize length of stay
        def quantizeLOS(x):
            if x<= 7:
                return '<week'
            if 8<x<=93:
                return '<3months'
            else:
                return '>3 months'

        # Quantize length of stay
        def adjustAge(x):
            if x == '25 - 45':
                return '25 to 45'
            else:
                return x

        # Quantize score_text to MediumHigh
        def quantizeScore(x):
            if (x == 'High')| (x == 'Medium'):
                return 'MediumHigh'
            else:
                return x

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        dfcutQ['priors_count_cat'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
        dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
        dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

        # Recode sex and race
        dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
        dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))

        features = ['two_year_recid',
                    'sex', 'race',
                    'age_cat', 'priors_count_cat', 'c_charge_degree', 'priors_count', 'length_of_stay']

        # Pass vallue to df
        df = dfcutQ[features]

        return df

    XD_features = ['age_cat', 'c_charge_degree', 'priors_count_cat', 'sex', 'race', 'priors_count', 'length_of_stay']
    D_features = ['sex', 'race']  if protected_attributes is None else protected_attributes
    Y_features = ['two_year_recid']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['age_cat', 'priors_count_cat', 'c_charge_degree']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0.0: 'Male', 1.0: 'Female'},
                                    "race": {1.0: 'Caucasian', 0.0: 'Not Caucasian'}}



    return CompasDataset(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=[],
        metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def load_preproc_data_german(protected_attributes=None):
    """
    Load and pre-process german credit dataset.
    Args:
        protected_attributes(list or None): If None use all possible protected
            attributes, else subset the protected attributes to the list.

    Returns:
        GermanDataset: An instance of GermanDataset with required pre-processing.

    """
    def custom_preprocessing(df):
        """ Custom pre-processing for German Credit Data
        """

        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                    'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)


        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = df['age'].apply(lambda x: np.int(x >= 26))
        df['status'] = df['status'].apply(lambda x: group_status(x))

        return df

    # Feature partitions
    XD_features = ['credit_history', 'savings', 'employment', 'sex', 'age', 'month', 'credit_amount']
    D_features = ['sex', 'age'] if protected_attributes is None else protected_attributes
    Y_features = ['credit']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['credit_history', 'savings', 'employment']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "age": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "age": {1.0: 'Old', 0.0: 'Young'}}
    return GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)



def preprocess_AIF_data(data_name, sensi_col, dense_outputs=True, output_path='../data/processed/'):
    # To preprocess Adult, German, and COMPAS using above AIF 360 preprocessing steps,
    # the skip cols below are not used in AIF 360 preprocessing but need to be presented for SingleCC and MultiCC
    if data_name == 'adult':
        dataset_orig = load_preproc_data_adult([sensi_col])
        skip_cols = ['age', 'hours-per-week']
    elif data_name == 'german':
        dataset_orig = load_preproc_data_german([sensi_col])
        skip_cols = ['month', 'credit_amount']
    elif data_name == 'compas':
        dataset_orig = load_preproc_data_compas([sensi_col])
        skip_cols = ['priors_count', 'length_of_stay']
    else:
        raise ValueError('Do not support the preprocessiong of the input dataset!')

    y_col = dataset_orig.label_names[0]
    ml_features = [x for x in dataset_orig.feature_names if x not in skip_cols]
    df_features, _ = dataset_orig.convert_to_dataframe()
    df_features.drop(columns=skip_cols, axis=1, inplace=True)
    # make sure class 1 represents positive outcome i.e., y=1.0 for positive outcome, 0.0 for negative
    if data_name == 'german':
        df_features['credit'] = df_features['credit'].apply(lambda x: float(x == 1.0))

    if data_name == 'compas':
        df_features['two_year_recid'] = df_features['two_year_recid'].apply(lambda x: float(x != 1.0))

    df_features = df_features.reindex(columns=ml_features + [y_col])
    df_features.to_csv(output_path + data_name + '_features.csv', index=False)
    # save a variant of the dataset that includes numerical attributes for SingleCC and MultiCC
    if dense_outputs:
        # keep the original categorical features for density estimation
        df_dense, _ = dataset_orig.convert_to_dataframe(de_dummy_code=True)
        if data_name == 'adult':
            dedummy_code = {'sex': {'Male': 1, 'Female': 0},
                            'Income Binary': {'>50K': 1, '<=50K': 0},
                            'Education Years': {'>12': 8, '12': 7, '11': 6, '10': 5, '9': 4, '8': 3, '7': 2, '6': 1,
                                                '<6': 0},
                            }
            output_kips = ['Age (decade)']
        elif data_name == 'german':
            dedummy_code = {'age': {'Old': 1, 'Young': 0},
                            'credit': {'Good Credit': 1, 'Bad Credit': 0},
                            'credit_history': {'None/Paid': 2, 'Other': 1, 'Delay': 0},
                            'savings': {'500+': 2, '<500': 1, 'Unknown/None': 0},
                            'employment': {'4+ years': 2, '1-4 years': 1, 'Unemployed': 0}}
            output_kips = None
        else: # compas
            dedummy_code = {'race': {'Caucasian': 1, 'Not Caucasian': 0},
                            'priors_count_cat': {'0': 2, '1 to 3': 1, 'More than 3': 0},
                            'two_year_recid': {'No recid.': 1, 'Did recid.': 0},
                            'age_cat': {'Greater than 45': 2, '25 to 45': 1, 'Less than 25': 0},
                            'c_charge_degree': {'M': 1, 'F': 0}
                            }
            output_kips = None

        for col, new_values in dedummy_code.items():
            df_dense[col] = df_dense[col].replace(new_values)
        if output_kips is not None:
            output_atts = list(set(df_dense.columns) - set(output_kips))
        else:
            output_atts = list(df_dense.columns)

        df_dense[output_atts].to_csv(output_path + data_name + '_dense.csv', index=False)

if __name__ == '__main__':
    processed_file_path = '../data/processed/'
    datasets = ['adult', 'german', 'compas']
    sensi_cols = ['sex', 'age', 'race']
    y_cols = ['Income Binary', 'credit', 'two_year_recid']

    for data_name, sensi_col in zip(datasets, sensi_cols):
        preprocess_AIF_data(data_name, sensi_col, output_path=processed_file_path)
