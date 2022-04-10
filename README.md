# The xtracker  project #

Neural network based trackfinding for Belle II experiment at SuperKEKB

### What is this repository for? ###

SuperKEKB is an asymmetric electron positron collider with a center of mass 
energy at 10.58 GeV, just at the mass of the Y(4s) resonance. Belle II is a 
hermetic detector build around the interaction point of SuperKEKB to record 
neutral and charged particles emerging from electron positron collisions. 
The tracking subdetectors of Belle II record hits of charged particles along 
their helical trajectory in a 1.5T solenoid magnetic field. Trackfinding is
the alogrithm that infers the charged particle from the observed pattern of 
hits. This repository implements a neural network based trackfinding method
usable as a new trackfinder the Belle II software (basf2).

The basf2 software allows for lightweigth extensions by adding reconstruction 
modules written in pure python. This feature allows rapid prototyping for machine 
learning applications since Torch and Tensorflow are readily included in basf2 
externals. The xtracker project adds additional basf2 reconstruction modules 
which read hits from tracking detectors, process them with neural nets and 
create tracks from hits as output. The hits are read from the basf2 DataStore 
and tracks are written back to the basf2 DataStore. In other words: we have 
realistic inputs (real data or full simulation) and can provide valid tracks 
for further downstream processing, like track fitting or vertex fitting or 
decay reconstruction. 

Nothing is for free: The prize is a reduced execution speed from using python 
modules. While this may be acceptable for an exploratory study, a final solution 
needs to be speed optimized. The main goal of this repository is to allow quick 
experimentation with different neural network architectures to find a optimal 
network architecture for a 'final' solution. Well knowing that nothing is really 
final in the machine learning space and openess is a true value. 

Note: Presently (March 2022), xtracker is only supported on the upgrade branch 
of the basf2 repository. The supported geometry is that of an all silicon pixel 
detector (5 or 7 layers )followed by the Central Drift Chamber and the outer 
Belle II subdetetors. 

Note: Presently, a neural network tracking based on graph neural networks (GNN) 
is available. This tracker is heavily inspired by ideas presented at the Connecting 
The Dots (CDT) conference 2018 by Steven Farrell (https://arxiv.org/abs/1810.06111)

### First time installation ###

In order to run the code a local installation of basf2 is needed. See the 
documentation at: https://b2-master.belle2.org/software/development/sphinx/index.html

0. Follow the instructions for a full local installation of basf2. The steps to follow are: 

```
mkdir b2
cd b2 
git clone ssh://git@stash.desy.de:7999/b2/tools.git
tools/b2install-prepare
```

This step needs to be executed only once for the first installation. `b2` will be the root 
directory for the Belle II software with xtracker added. 

1. Once the tools are installed you need to setup the Belle II environment by sourcing 

```
source tools/b2setup
```

2. Create the local installation in a folder called development: 

```
b2code-create development
```

3. Change into the development directory, switch to the upgrade branch and build basf2: 

```
cd development
git checkout upgrade
b2setup
scons -j 4
```

At the first installation, `b2setup` will tell you that external software dependencies 
are missing. You will be adviced to `b2install-externals` to install the a certain 
version of the externals. Presently it would be `v01-10-02`. For many operating system, 
there are pre-compiled binaries available. For Ubuntu 20.04, you can use: 

```
b2install-externals v01-10-02 ubuntu2004
```

4. For xtracker, we need to install torch-scatter on top of the externals. 


```
python3 -m pip install  torch-scatter -f https://data.pyg.org/whl/torch-1.8.1+cpu.html
```


5. And finally we can clone xtracker and install it locally:

```
cd b2
git clone https://github.com/BenjaminSchwenker/xtracker.git
cd xtracker
python3 setup.py develop
```

Done.

### How to setup the environment, after first time installation done ###

```
cd <b2>
source tools/b2setup
cd development
b2setup
cd ../xtracker
```

`<b2>` is the path to the folder `b2` where your basf2 installation resides. Now you are 
ready to explore the examples folder or start your own development. 

### First steps ###

The example folder contains the main python scripts needed to train a tracking net 
from scratch, evaluate it using the standard Belle II tracking validation methods 
and reconstruct a sample of simualed Belle II MC events. 


1. Simulate a MC sample of events for training

Please see [https://confluence.desy.de/pages/viewpage.action?spaceKey=BI&title=Full+simulation+effort](Full Simulation Effort) 
for an up to date description of options for geometry options and locactions of background overlay files. 

In order to simulate an upgraded Belle II detector with a 5 layer VTX vertex detector w/o backgrounds run: 

```
cd xtracker/examples
export BELLE2_VTX_UPGRADE_GT=upgrade_2022-01-21_vtx_5layer
export BELLE2_VTX_BACKGROUND_DIR=None
export HEP_DATA=<some/path/with/storage>
basf2 simulate_belle2.py -- configs/belle2_vtx.yaml
```

Environment variables are used to define the conditions data (GT), location of overlay files and path of 
output data. The output data consists of a folder under ${HEP_DATA} with one file per event containing
the hit data of Belle II tracking detectors, in a non Belle II specific data fromat (HDF5). 

2. Create event graphs from event data

```
python3 prepare_graphs.py configs/belle2_vtx.yaml --n-workers=3
```

It creates a folder under ${HEP_DATA} with one hitgraph file (.npz) per event as input for training.

3. Train graph neural network on sample of event graphs 

```
python3 train.py  configs/belle2_vtx.yaml
```

4. Validate neural network tracking on independent events 

In order to run the tracking validation, some validation scripts need to be copied over to 
the tracking/validation folder of your basf2 installation. 

```
cp <b2>/xtracker/examples/validation_scripts/*.py <b2>/development/tracking/validation/
export XTRACKER_CONFIG_PATH=<b2>/xtracker/examples/configs/configfile.yaml
```

Now we can simple use the `b2validation` command to execute a validation run. The environment 
variable `XTRACKER_CONFIG_PATH` points to your model config to tell the script where the saved
model resides.  


```
export BELLE2_VTX_UPGRADE_GT=upgrade_2022-01-21_vtx_5layer
export BELLE2_VTX_BACKGROUND_DIR=/path/to/bgfiles/
b2validation -s  upgradeVTXOnlyTrackingValidation.py upgradeVTXOnlyTrackingValidationBkg.py 
```

You can add the option `-o '-n 20'` which forces basf2 to validate on 20
events. This is good for fast testing. A more reasonable number is 1000 events.  

5. 

```
basf2 test_belle2.py -n 10 -- configs/belle2_vtx.yaml
```


### Who do I talk to? ###

If you should stumble accross this project and have questions, feel free to contact 
benjamin.schwenker@phys.uni-goettingen.de

### Contributing ###

Please also see CONTRIBUTING.md. I am happy for any new idea for trying neural network based 
tracking for Belle II tracking or track based triggering. 

- Please use the integrated issue tracker at gitlab to report problems or suggest new features. 

- Please use the issue tracker also for creating new branches to implement bugfixes or new features. 

- Please add benjamin.schwenker as reviewer to your pull requests into the master branch. 
