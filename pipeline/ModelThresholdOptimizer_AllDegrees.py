# Select the optimal classification threshold using validation set for LR models for multiple intervention degrees of SingleCC
import warnings
import argparse
from multiprocessing import Pool, cpu_count

from ModelThresholdOptimizer import compute_model_optimal_thresh

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find optimal threshold on validation set for LR models")

    parser.add_argument("--setting", type=str,
                        help="setting of SingleCC. When is not None, called after SingleCC. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()


    intervention_scales = [(0.1+x/10, 0.04+3*x/100) for x in range(20)]
    datasets = ['meps16', 'lawgpa']
    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
                                                                0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'
    opt_obj = 'BalAcc'

    if args.setting is None:
        raise ValueError('The input "setting" is requried. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    if args.setting not in ['SingleCC', 'KAM-CAL', 'SingleCC+KAM-CAL']:
        raise ValueError('The input "setting" is not supported. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for weight_i in intervention_scales:
                weight_suffix = '{}_{}'.format(weight_i[0], weight_i[1])
                for seed in seeds:
                    tasks.append([data_name, seed, args.setting, res_path, opt_obj, weight_suffix])
        with Pool(cpu_count()) as pool:
            pool.starmap(compute_model_optimal_thresh, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')
