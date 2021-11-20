# xtracker

Neural network based trackfinding for high energy physics collider

## Contents 

The example folder contains the main python scripts for running:

- *[prepare.py](prepare.py)*: the data preparation script which simulates 
Belle II MC events, cleans and reduces the data, and writes hit graphs to
the filesystem.
- *[train.py](train.py)*: the main training script which is steered by
configuration file and loads the data, model, and trainer, and invokes
the trainer to train the model.
- *[example.py](example.py)*: an example script that simulates Belle II MC
events and uses the trained model for reconstruction. 



## How do I get set up? 

In order to run the code a local installation of basf2 is needed. See the 
documentation at: https://b2-master.belle2.org/software/development/sphinx/index.html

1. Setup your local basf2 environment. You can use your local environment (installed on your machine) of a release on cvmfs. From 

```
cd release-directory
source tools-directory/b2setup
```

2. Install torch_scatter from pipy into your environment

a. If you have a local installation, you can use the normal setup command

```
python3 -m pip install  torch-scatter -f https://data.pyg.org/whl/torch-1.8.1+cpu.html
python3 -m pip install torch-sparse -f https://data.pyg.org/whl/torch-1.8.1+cpu.html
python3 -m pip install torch-geometric
```

## Who do I talk to?

If you should stumble accross this project and have questions, feel free to contact 
benjamin.schwenker@phys.uni-goettingen.de
