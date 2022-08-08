
# the below scripts can run systematically
python ModelTrainer.py --set_n -1 --exec_n 1
python ModelThresholdOptimizer.py --model 'single' --setting 'SingleCC' --set_n -1 --exec_n 1
python ModelThresholdOptimizer.py --model 'group' --set_n -1 --exec_n 1

# optimize CC and learn rules using CCs
python CCModelTrainer.py --set_n -1 --exec_n 1
python FairInterMulti.py --setting 'orig' --set_n -1 --exec_n 1
python ModelEvaluator_FairInterMulti.py --set_n -1 --exec_n 1

# apply fairness interventions on training data using SingleCC, KAM-CAL, and SingleCC+KAM-CAL
python FairInterSingle.py --setting 'SingleCC' --set_n -1 --exec_n 1
python FairInterSingle.py --setting 'KAM-CAL' --set_n -1 --exec_n 1
python FairInterSingle.py --setting 'SingleCC+KAM-CAL' --set_n -1 --exec_n 1

python ModelThresholdOptimizer.py --model 'single' --setting 'SingleCC' --set_n -1 --exec_n 1
python ModelThresholdOptimizer.py --model 'single' --setting 'KAM-CAL' --set_n -1 --exec_n 1
python ModelThresholdOptimizer.py --model 'single' --setting 'SingleCC+KAM-CAL' --set_n -1 --exec_n 1

python ModelEvaluator_FairInterSingle.py --setting 'SingleCC' --set_n -1 --exec_n 1
python ModelEvaluator_FairInterSingle.py --setting 'KAM-CAL' --set_n -1 --exec_n 1
python ModelEvaluator_FairInterSingle.py --setting 'SingleCC+KAM-CAL' --set_n -1 --exec_n 1


# compare SingleCC, KAM-CAL, and CAPUCHIN using XGBoost Tree
python ModelTrainer_CAP.py --set_n -1 --exec_n 1
python ModelTrainer_NonInvasive.py --setting 'SingleCC' --set_n -1 --exec_n 1
python ModelTrainer_NonInvasive.py --setting 'KAM-CAL' --set_n -1 --exec_n 1
python ModelThresholdOptimizer_NonInvasive.py  --setting 'SingleCC_xgb' --set_n -1 --exec_n 1
python ModelThresholdOptimizer_NonInvasive.py  --setting 'KAM-CAL_xgb' --set_n -1 --exec_n 1
python ModelEvaluator_FairInterSingle_XGB.py --set_n -1 --exec_n 1


# special experiments for erroneous test data
# under fixed error rate
python ErrorDataSimulator.py --setting 'error0.15' --set_n -1 --exec_n 1
python FairInterMulti.py --setting 'error0.15' --set_n -1 --exec_n 1
python ModelEvaluator_Error.py --data 'UFRGS' --model 'group' --setting 'error0.15' --exec_n 1
python ModelEvaluator_Error.py --data 'UFRGS' --model 'single' --setting 'error0.15' --exec_n 1

# multiple error rates
python ErrorDataSimulator_AllRates.py --data 'UFRGS' --exec_n 1 --exec_k 1
python FairInterMulti_AllErrorRates.py --data 'UFRGS' --exec_n 1 --exec_k 1
python ModelEvaluator_Error.py --data 'UFRGS' --model 'group' --setting 'all' --exec_n 1 --exec_k 1


# special experiments for multiple intervention degrees in SingleCC
python FairInterSingle_AllDegrees.py --data 'UFRGS' --setting 'SingleCC' --exec_n 1 --exec_i 1
python FairInterSingle_AllDegrees.py --data 'UFRGS' --setting 'SingleCC+KAM-CAL' --exec_n 1 --exec_i 1

python ModelThresholdOptimizer_AllDegrees.py --data 'UFRGS' --setting 'SingleCC' --exec_n 1 --exec_i 1
python ModelThresholdOptimizer_AllDegrees.py --data 'UFRGS' --setting 'SingleCC+KAM-CAL' --exec_n 1 --exec_i 1

python ModelEvaluator_AllDegrees.py --data 'UFRGS' --setting 'SingleCC' --exec_n 1 --exec_i 1
python ModelEvaluator_AllDegrees.py --data 'UFRGS' --setting 'SingleCC+KAM-CAL' --exec_n 1 --exec_i 1



