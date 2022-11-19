python RetrainModelwithWeights.py --weight 'scc' # only seed 1
python RetrainModelwithWeights.py --weight 'scc' --base 'omn' # only seed 1


python RetrainModelwithWeights.py --weight 'scc' --base 'kam'
python RetrainModelwithWeights.py --weight 'kam'
python RetrainModelwithWeights.py --weight 'omn'


python TuneWeightScales.py --weight 'omn'
python TuneWeightScales.py --weight 'scc'
python TuneWeightScales.py --weight 'scc' --base 'kam'
python TuneWeightScales.py --weight 'scc' --base 'omn'

python TrainMLModels.py --set_n 1 --exec_n 1

python LearnCCrules.py --set_n 1 --exec_n 1

python RepairDataByCAP.py

python RetrainTreeModelsWithWeights.py --weight 'scc' --base 'kam' --high 0.1 --set_n 1 --exec_n 1