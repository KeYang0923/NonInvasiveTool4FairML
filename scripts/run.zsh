python RetrainModelwithWeights.py --weight 'scc' # only seed 1
python RetrainModelwithWeights.py --weight 'scc' --base 'omn' # only seed 1


python RetrainModelwithWeights.py --weight 'scc' --base 'kam'
python RetrainModelwithWeights.py --weight 'kam'
python RetrainModelwithWeights.py --weight 'omn'


python TuneWeightScales.py --weight 'omn'
python TuneWeightScales.py --weight 'scc'
python TuneWeightScales.py --weight 'scc' --base 'kam'
python TuneWeightScales.py --weight 'scc' --base 'omn'