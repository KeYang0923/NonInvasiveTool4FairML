# Use fairness interventions on training data with a single model and retrain the same ML models multiple interention degrees
import warnings

import argparse
from multiprocessing import Pool, cpu_count
from FairInterSingle import retrain_models_by_faircc
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply fairness intervention on training data with a single ML model and retrain the model under multiple interention degrees")
    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()

    intervention_scales = [(0.1 + x / 10, 0.04 + 3 * x / 100) for x in range(20)]
    datasets = ['meps16', 'lawgpa']
    y_cols = ['Y' for i in range(2)]
    sensi_cols = ['C0' for i in range(2)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    if args.setting not in ['SingleCC', 'KAM-CAL', 'SingleCC+KAM-CAL']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    res_path = '../intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for weight_i in intervention_scales:
                weight_suffix = '{}_{}'.format(weight_i[0], weight_i[1])
                cur_adjust = {data_name: weight_i}
                for seed in seeds:
                    tasks.append([data_name, seed, y_col, sensi_col, args.setting, cur_adjust, res_path, weight_suffix])

        with Pool(cpu_count()) as pool:
            pool.starmap(retrain_models_by_faircc, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')
