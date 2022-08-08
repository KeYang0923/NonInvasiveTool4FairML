# Use fairness interventions on training data with a single model and retrain the same ML models multiple interention degrees
import warnings

import argparse
from multiprocessing import Pool, cpu_count
from FairInterSingle import retrain_models_by_faircc
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply fairness intervention on training data with a single ML model and retrain the model under multiple interention degrees")
    parser.add_argument("--data", type=str,
                        help="dataset to simulate all the intervention degree. Use 'all' for all the datasets. OR choose from [meps16, lawgpa, UFRGS] for different datasets.")

    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running few number of executions and few intervention degrees
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    parser.add_argument("--exec_i", type=int, default=20,
                        help="number of intervention degrees. Default is 20.")
    args = parser.parse_args()


    datasets = ['meps16', 'lawgpa', 'UFRGS']

    if args.data is None:
        raise ValueError('The input "data" is requried. Use "all" for all the datasets. OR choose from [meps16, lawgpa, UFRGS] for different datasets.')

    elif args.data == 'all':
        pass
    else:
        if args.data not in datasets:
            raise ValueError(
                'The input "data" is not supported. Choose from [meps16, lawgpa, UFRGS] for different datasets.')
        else:
            datasets = [args.data]
            y_cols = ['Y']
            sensi_cols = ['C0']

    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    if args.setting not in ['SingleCC', 'KAM-CAL', 'SingleCC+KAM-CAL']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    intervention_scales = [(0.1 + x / 10, 0.04 + 3 * x / 100) for x in range(20)]
    if args.exec_i is None:
        raise ValueError(
            'The input "exec_i" is requried. Use "--exec_i 1" for a single intervention degree.')
    elif type(args.exec_i) == str:
        raise ValueError(
            'The input "exec_i" requires integer. Use "--exec_i 1" for a single intervention degree.')
    else:
        i_exec = int(args.exec_i)
        intervention_scales = intervention_scales[:i_exec]


    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

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
