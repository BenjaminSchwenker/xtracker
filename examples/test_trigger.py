#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

"""
Simple script to generate a sample of Belle II events and study
the trigger performance.

Usage: Set global tag and run
export BELLE2_VTX_UPGRADE_GT=upgrade_2022-01-21_vtx_5layer
export BELLE2_VTX_BACKGROUND_DIR=/path/to/bgfiles/
export HEP_DATA=/path/to/trained/models

basf2 test_trigger.py -n 200 -- configs/belle2_vtx.yaml --gen GUN
"""

import basf2 as b2
import generators as ge
import simulation as si
import modularAnalysis as ma
from variables import variables as vm

import argparse
import os
import yaml
from xtracker.path_utils import add_vtx_trigger, add_trigger_EffCalculation
from vtx import add_vtx_reconstruction, get_upgrade_globaltag, get_upgrade_background_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('test_trigger.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/belle2_vtx.yaml')
    add_arg("--gen", default='bbar', help="Generator: 'GUN', 'GUND', 'BGONLY', 'BBBAR'")
    add_arg("--thr", default=0.7, type=float, help="Threshold for VTX trigger classifier output")
    add_arg("--output", default="trigger.root", help="Output file for ntuple")
    return parser.parse_args()


def main():
    """Main function"""

    # Need to use default global tag prepended with upgrade GT
    b2.conditions.disable_globaltag_replay()
    b2.conditions.prepend_globaltag(get_upgrade_globaltag())

    # Parse the command line
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create basf2 path
    path = b2.create_path()

    eventinfosetter = b2.register_module('EventInfoSetter')
    # default phase3 geometry:
    exp_number = 0
    eventinfosetter.param("expList", [exp_number])
    path.add_module(eventinfosetter)

    eventinfoprinter = b2.register_module('EventInfoPrinter')
    path.add_module(eventinfoprinter)

    progress = b2.register_module('Progress')
    path.add_module(progress)

    # simulation settings
    if args.gen == 'BBBAR':
        evtgenInput = b2.register_module('EvtGenInput')
        evtgenInput.logging.log_level = b2.LogLevel.WARNING
        path.add_module(evtgenInput)
    elif args.gen == 'GUN':
        path.add_module(
            'ParticleGun',
            nTracks=1,
            pdgCodes=[-211, 211],
            momentumParams=[0.05, 5.0],
            xVertexParams=[0.0],
            yVertexParams=[0.0],
            zVertexParams=[0.0]
        )
    elif args.gen == 'GUND':
        path.add_module(
            'ParticleGun',
            nTracks=1,
            pdgCodes=[-211, 211],
            momentumParams=[0.05, 4.0],
            vertexGeneration='uniform',
            xVertexParams=[- 0.005, + 0.005],
            yVertexParams=[- 0.005, + 0.005],
            zVertexParams=[- 12, + 12],
        )
    elif args.gen == "BHABHA":
        ge.add_babayaganlo_generator(path=path, finalstate='ee', minenergy=0.15, minangle=10.0)
    elif args.gen == "MUMU":
        ge.add_kkmc_generator(path=path, finalstate='mu+mu-')
    elif args.gen == "EEEE":
        ge.add_aafh_generator(path=path, finalstate='e+e-e+e-', preselection=False)
    elif args.gen == "EEMUMU":
        ge.add_aafh_generator(path=path, finalstate='e+e-mu+mu-', preselection=False)
    elif args.gen == "TAUPAIR":
        ge.add_kkmc_generator(path, finalstate='tau+tau-')
    elif args.gen == "DDBAR":
        ge.add_continuum_generator(path, finalstate='ddbar')
    elif args.gen == "UUBAR":
        ge.add_continuum_generator(path, finalstate='uubar')
    elif args.gen == "SSBAR":
        ge.add_continuum_generator(path, finalstate='ssbar')
    elif args.gen == "CCBAR":
        ge.add_continuum_generator(path, finalstate='ccbar')
    elif args.gen == 'BGONLY':
        # shoot a single extremely low momentum pion to make T0 simulation happy
        path.add_module(
            'ParticleGun',
            pdgCodes=[-211, 211],
            momentumParams=[0.001, 0.002],
            xVertexParams=[0.0],
            yVertexParams=[0.0],
            zVertexParams=[0.0]
        )

    # Detector Simulation:
    si.add_simulation(path=path, bkgfiles=get_upgrade_background_files(), useVTX=True)

    # VTX reconstruction
    add_vtx_reconstruction(path=path)

    add_vtx_trigger(
        path=path,
        tracker_model_path=config['training']['checkpoint'],
        trigger_model_path=os.path.expandvars("${HEP_DATA}/model_vtx_trigger/"),
        event_cuts=config['event_cuts'],
        segment_cuts=config['selection'],
        tracker_config=config['model'],
        threshold=args.thr,
    )

    add_trigger_EffCalculation(path)

    vm.addAlias('l1', 'L1Trigger')
    vm.addAlias('hie', 'L1PSNM(hie)')
    vm.addAlias('c4', 'L1PSNM(c4)')
    vm.addAlias('eclmumu', 'L1PSNM(eclmumu)')
    vm.addAlias('ffo', 'L1PSNM(ffo)')
    vm.addAlias('fff', 'L1PSNM(fff)')
    vm.addAlias('fso', 'L1PSNM(fso)')
    vm.addAlias('sso', 'L1PSNM(sso)')
    vm.addAlias('ffs', 'L1PSNM(ffs)')
    vm.addAlias('sss', 'L1PSNM(sss)')
    vm.addAlias('vtx',  'eventExtraInfo(VTXTrigger)')
    vm.addAlias('vtx_out',  'eventExtraInfo(VTXTriggerClassifierOutput)')

    my_event_variables = [
        'l1',
        'hie',
        'c4',
        'eclmumu',
        'ffo',
        'fff',
        'fso',
        'sso',
        'ffs',
        'sss',
        'vtx',
        'vtx_out'
    ]

    ma.fillParticleListFromMC('pi+:MC', cut='', addDaughters=True, skipNonPrimaryDaughters=True, path=path)

    path.add_module('VariablesToEventBasedTree',
                    fileName=args.output,
                    particleList='pi+:MC',
                    variables=['isSignal', 'p', 'pt', 'pz', 'E', 'z0', 'phi', 'theta', 'charge'],
                    event_variables=my_event_variables
                    )

    # add output
    # path.add_module('RootOutput')

    b2.print_path(path)

    b2.process(path)
    print(b2.statistics)


if __name__ == '__main__':
    main()
