python DataPreprocessor.py --data 'UFRGS'
python ModelTrainer.py --set_n -1
python CCModelTrainer.py --set_n -1

python ModelThresholdOptimizer.py --model 'group' --set_n -1
python FairInterMulti.py --setting 'orig' --set_n -1
python ModelEvaluator_FairInterMulti.py --set_n -1

python FairInterSingle.py --setting 'SingleCC' --set_n -1
python ModelThresholdOptimizer.py --model 'single' --setting 'SingleCC' --set_n -1
python ModelEvaluator_FairInterSingle.py --setting 'SingleCC' --set_n -1