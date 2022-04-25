#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# basf2 (Belle II Analysis Software Framework)                           #
# Author: The Belle II Collaboration                                     #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

from basf2 import Module
from ROOT import Belle2


class EffModule(Module):
    """
    This module is to calculate and print out the efficiency of each L1 trigger line with
    the trigger result from object 'TRGGDLResults'
    """
    #: The total number of events
    Ntot_event = 0
    #: The number of events passing L1 trigger
    Ntrg_event = 0
    #: The number of events passing each L1 trigger line
    Nsubtrg_event = [0]
    #: prescale factors for phase3
    prescale_phase3 = [1]

    #: trigger bit log for phase3
    trglog_phase3 = [
                     'VTX: VTX trigger classifier output > thr'
                     ]

    # ---add new trigger line by users---
    # ---add a component with initial value 0 in Nsubtrg_event
    # Nsubtrg_event+=[0]
    # ---add the prescale factor in prescale list
    # prescale += [1]
    # ---add the description of new trigger logics in trglog
    # trglog+=['new trg logics']

    def __init__(self):
        """Initialization of EffModule"""
        super(EffModule, self).__init__()

    def event(self):
        """
        Event function to count the numbers of events passing each trigger line
        """
        self.Ntot_event += 1
        trgresult = Belle2.PyStoreObj('TRGSummary')
        summary = trgresult.getPsnmBits(0)
        if summary >= 1:
            self.Ntrg_event += 1

        # Logic for VTX
        vtxtrg = Belle2.PyStoreObj("EventExtraInfo").getExtraInfo("VTXTrigger")
        if vtxtrg == 1:
            self.Nsubtrg_event[-1] += 1

    def terminate(self):
        """
        Calculate the efficiency of each trigger line with the statistical values in event function
        """

        trglog = self.trglog_phase3
        prescale = self.prescale_phase3

        #: Total number of events
        if self.Ntot_event == 0:
            return
        sp = ' '
        print('\n')
        eff_tot = self.Ntrg_event / self.Ntot_event * 100.0
        print('L1 Trigger efficiency(%%): %6.4f' % (eff_tot))
        print('VTX Trigger Line', 5 * sp, 'PreScale Factor', 3 * sp, 'Efficiency(%)', 3 * sp, 'Logics')
        ntrg = len(self.Nsubtrg_event)
        if self.Ntot_event != 0:
            for i in range(ntrg):
                eff = self.Nsubtrg_event[i] / self.Ntot_event * 100.0
                print('T%3d                %4d              %6.4f              %s ' % (i, prescale[i], eff, trglog[i]))
