#!/bin/bash

export BELLE2_VTX_BACKGROUND_DIR=None


basf2 test_trigger.py -n 1000 -- configs/belle2_vtx.yaml --gen GUN --output Trigger_gun.root

basf2 test_trigger.py -n 1000 -- configs/belle2_vtx.yaml --gen GUND --output Trigger_gund.root

basf2 test_trigger.py -n 1000 -- configs/belle2_vtx.yaml --gen BBBAR --output Trigger_bbbar.root

# Adopt to your system
export BELLE2_VTX_BACKGROUND_DIR=/home/benjamin/b2/bgtest/vtx_5layer/


basf2 test_trigger.py -n 1000 -- configs/belle2_vtx.yaml --gen GUN --output Trigger_gun_bg.root

basf2 test_trigger.py -n 1000 -- configs/belle2_vtx.yaml --gen BGONLY --output Trigger_bg.root

basf2 test_trigger.py -n 1000 -- configs/belle2_vtx.yaml --gen BBBAR --output Trigger_bbbar_bg.root
