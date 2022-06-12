# PCGCE


Package to test causal discovery algorithm on simulated and real data 



## Methods

* PCTCE: 
* TiMINO: https://proceedings.neurips.cc/paper/2013/file/47d1e990583c9c67424d369f3414728e-Paper.pdf
* VarLiNGAM: https://www.jmlr.org/papers/volume11/hyvarinen10a/hyvarinen10a.pdf
* PCMCICMIknn: http://proceedings.mlr.press/v124/runge20a/runge20a.pdf
* oCSE: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.474.6986&rep=rep1&type=pdf
* tsFCI: 
* DyNoTears: https://arxiv.org/pdf/2002.00498.pdf
* GangerLasso: 
* TCDF: 

Some algorithms are imported from other langauges such as R and Java


## Test

### To test algorithms on simulated data run:
python3 test_simulated_data_v2.py method structure graph_type n_samples num_processor verbose

* method: causal dicovery algorithms, choose from [PCTCE, GangerLasso, TCDF, PCMCIplusCMIknn, oCSE, tsFCI, VarLiNGAM, TiMINO, Dynotears]
* structure: causal structure, choose from [diamond, cycle, 7ts2h]
* graph_type: choose from [acyclic, cyclic]
* self_cause: chose from [True, False] 
* max_lag: maximal lag
* n_samples: number of timestamps
* num_processor: number of processors

Example: python3 test_fmri.py "PCTCE" "fork" 1000 1 1

### To test algorithms on fmri data run:
python3 test_simulated_data.py method num_processor verbose

Example: python3 test_fmri.py "PCTCE" 1 1
