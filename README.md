# JAMPR+ LNS

This repo implements the Large Neighborhood Search based on
Neural Construction Heuristics of the corresponding [paper](https://arxiv.org/abs/2205.00772).

---
It is evaluated on the DIMACS [VRP-TW track](http://dimacs.rutgers.edu/programs/challenge/vrp/vrptw/)
and is a 
- reworked and improved version of the original [JAMPR model](https://software.ismll.uni-hildesheim.de/ISMLL-internal/TrAmP)
- using the JAMPR model as a neural construction heuristic
- in combination with simple destruction heuristics and an additional local search routine
- in a Large Neighborhood Search framework.


Please cite us: 
```
@article{falkner2022large,
  title={Large Neighborhood Search based on Neural Construction Heuristics},
  author={Falkner, Jonas K and Thyssens, Daniela and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:2205.00772},
  year={2022}
}
``` 

### Setup
Install the requirements as conda environment
```sh
conda env create -f requirements.yml
```


### Training
Train JAMPR for tsptw
```sh
# training
python run_jampr.py meta=train env=tsptw
# test
python run_jampr.py meta=eval .....?
```

### LNS
Run the LNS on the Solomon instance c103 with a time limit of 60s
```sh
python run_lns.py data_path=./data/solomon_txt/c1/c103.txt time_limit=60
```

All runs are configured via hydra through the config files in the 
[config](./config) and [config_lns](./config_lns) directories.
A configuration summary can also be found via 
```sh
python run_jampr.py -h
```
and
```sh
python run_lns.py -h
```


### VRP Challenge

Run the controller with LNS for instance R101 (override time limit for testing).
Requires a corresponding customized controller from the DIMACS challenge!
```sh
python run_controller.py --id R101 --time_limit_override 20
```
