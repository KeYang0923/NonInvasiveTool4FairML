# prepare data for model training, execute this zash inside the pipeline folder
python DataPreprocessor_AIF.py # for benchmark datasets Adult income, German credit, and COMPAS
python DataProcessor.py --data 'all' # for other real datasets Cardio diseases, Bank marketing, MEPS, Law School GPA, Credit, and UFRGS

# running CAPUCHIN to repair data, serial execution
cd ../CAPUCHIN/
python DataProcessor_CAP.py

# the below scripts can run systematically
cd ../pipeline/
python ModelTrainer.py
python ModelThresholdOptimizer.py --model 'single' --setting 'SingleCC'
python ModelThresholdOptimizer.py --model 'group'


# optimize CC and learn rules using CCs
python CCModelTrainer.py
python FairInterMulti.py --setting 'orig'
python ModelEvaluator_FairInterMulti.py

# apply fairness interventions on training data using SingleCC, KAM-CAL, and SingleCC+KAM-CAL
python FairInterSingle.py --setting 'SingleCC'
python FairInterSingle.py --setting 'KAM-CAL'
python FairInterSingle.py --setting 'SingleCC+KAM-CAL'

python ModelThresholdOptimizer.py --model 'single' --setting 'SingleCC'
python ModelThresholdOptimizer.py --model 'single' --setting 'KAM-CAL'
python ModelThresholdOptimizer.py --model 'single' --setting 'SingleCC+KAM-CAL'

python ModelEvaluator_FairInterSingle.py --setting 'SingleCC'
python ModelEvaluator_FairInterSingle.py --setting 'KAM-CAL'
python ModelEvaluator_FairInterSingle.py --setting 'SingleCC+KAM-CAL'


# compare SingleCC, KAM-CAL, and CAPUCHIN using XGBoost Tree
python ModelTrainer_CAP.py
python ModelTrainer_NonInvasive.py --setting 'SingleCC'
python ModelTrainer_NonInvasive.py --setting 'KAM-CAL'
python ModelThresholdOptimizer_NonInvasive.py  --setting 'SingleCC_xgb'
python ModelThresholdOptimizer_NonInvasive.py  --setting 'KAM-CAL_xgb'
python ModelEvaluator_FairInterSingle_XGB.py


# special experiments for erroneous test data
# under fixed error rate
python ErrorDataSimulator.py --setting 'error0.15'
python FairInterMulti.py --setting 'error0.15'
python ModelEvaluator_Error.py --data 'all' --model 'group' --setting 'error0.15'
python ModelEvaluator_Error.py --data 'all' --model 'single' --setting 'error0.15'

# multiple error rates
python ErrorDataSimulator_AllRates.py --data 'lawgpa'
python FairInterMulti_AllErrorRates.py --data 'lawgpa'
python ModelEvaluator_Error.py --data 'lawgpa' --model 'group' --setting 'all'


# special experiments for multiple intervention degrees in SingleCC
python FairInterSingle_AllDegrees.py --setting 'SingleCC'
python FairInterSingle_AllDegrees.py --setting 'SingleCC+KAM-CAL'

python ModelThresholdOptimizer_AllDegrees.py --setting 'SingleCC'
python ModelThresholdOptimizer_AllDegrees.py --setting 'SingleCC+KAM-CAL'

python ModelEvaluator_AllDegrees.py --setting 'SingleCC'
python ModelEvaluator_AllDegrees.py --setting 'SingleCC+KAM-CAL'



