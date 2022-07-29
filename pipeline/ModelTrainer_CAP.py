# Train and store XGBoost Tree models on repaired data from CAPUCHIN
import warnings
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from utils import make_folder, split
from ModelTrainer import Learner
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from joblib import dump

warnings.filterwarnings(action='ignore')

class XgBoost(Learner):

    def __init__(self, scoring='accuracy'):
        super(XgBoost, self).__init__(scoring)
        self.name = "XGB"

    def fit(self, train_data, train_Y, cat_cols, sample_weight=None):

        if self.scoring == 'accuracy':
            xg_metric = 'error'

        if self.scoring == 'roc_auc':
            xg_metric = 'auc'

        feature_transformation = ColumnTransformer(transformers=[
            ('encode_categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ])

        param_grid = {
            'learner__n_estimators': [5, 10],
            'learner__max_depth': [3, 6, 10],
            'learner__objective': ['binary:logistic'],
            'learner__eval_metric': [xg_metric]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', xgb.XGBClassifier())])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=-1, n_jobs=-1)

        if sample_weight is not None:
            model = search.fit(train_data, train_Y, **{'learner__sample_weight': sample_weight})
        else:
            model = search.fit(train_data, train_Y)
        return model



def XGB_trainer(data_name, y_col, sensi_col, seed, num_atts, repair_path='../intermediate/cap_res/',
                  res_path='../intermediate/models/',
                  verbose=False, data_path='../data/processed/'):

    cur_dir = res_path + data_name + '/'
    make_folder(cur_dir)

    train_df = pd.read_csv(repair_path + data_name + '/train__repMF_'+str(seed)+'.csv')

    if data_name in ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']:
        learner = XgBoost()
    else:
        raise ValueError('The input dataset is not supported!. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit]')

    if data_name in ['lawgpa', 'credit']:
        test_df = pd.read_csv(repair_path +data_name+'/test_cat'+str(seed)+'.csv')
    else:
        input_df = pd.read_csv(data_path + data_name + '_dense.csv')
        _, _, test_df = split(input_df, seed)

    test_df.drop(columns=num_atts, inplace=True)

    features = [x for x in train_df.columns if x != y_col and x != sensi_col]

    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    test_data = test_df[features]
    Y_test = np.array(test_df[y_col])

    model = learner.fit(train_data, Y_train, features)

    test_df['Y'] = Y_test
    train_df['Y'] = Y_train

    test_df['Y_pred'] = model.predict(test_data)
    train_df['Y_pred'] =  model.predict(train_data)


    dump(model, cur_dir + '-'.join(['model', 'CAP', str(seed)]) + '.joblib')

    train_df.to_csv(cur_dir + '-'.join(['train', 'CAP', str(seed)]) + '.csv') # keep index for sanity check of random splits
    test_df.to_csv(cur_dir + '-'.join(['test', 'CAP', str(seed)]) + '.csv', index=False)

    if verbose:
        score_train = learner.score(Y_train, np.array(train_df['Y_pred']))
        score_test = learner.score(Y_test, np.array(test_df['Y_pred']))
        print('---' * 8, data_name, seed, '---' * 8)
        print(learner.scoring, "on train data: ", score_train)
        print(learner.scoring, "on test data: ", score_test)
        print('---' * 10, '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XGBoost Tree models for repaired data from CAPUCHIN")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--setting", type=str, default='CAP',
                        help="input of CAPUCHIN. Default is CAPUCHIN only. Choose from [CAP, singleCAP].")

    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(5)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(5)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
             0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    num_atts_mapping = {'adult': ['hours-per-week', 'age'],
                        'german': ['month', 'credit_amount'],
                        'compas': ['priors_count', 'length_of_stay'],
                        'cardio': ['X' + str(i) for i in range(1, 4)],
                        'bank': ['X' + str(i) for i in range(1, 5)],
                        'meps16': ['X' + str(i) for i in range(1, 5)],
                        'lawgpa': ['X' + str(i) for i in range(1, 3)],
                        'credit': ['X' + str(i) for i in range(1, 6)]
                        }


    repair_path = '../intermediate/cap_res/'

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            num_atts = num_atts_mapping[data_name]
            for seed in seeds:
                tasks.append([data_name, y_col, sensi_col, seed, num_atts, repair_path])
        with Pool(cpu_count()) as pool:
            pool.starmap(XGB_trainer, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')