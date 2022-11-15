python ErrorDataSimulator.py --setting 'error0.30'
python FairInterMulti.py --setting 'error0.30'
python ModelEvaluator_Error.py --data 'all' --model 'group' --setting 'error0.30'
python ModelEvaluator_Error.py --data 'all' --model 'single' --setting 'error0.30'


python ErrorDataSimulator_AllRates.py --data 'meps16'
python FairInterMulti_AllErrorRates.py --data 'meps16'
python ModelEvaluator_Error.py --data 'meps16' --model 'group' --setting 'all'
python ErrorDataSimulator_AllRates.py --data 'lawgpa'
python FairInterMulti_AllErrorRates.py --data 'lawgpa'
python ModelEvaluator_Error.py --data 'lawgpa' --model 'group' --setting 'all'
