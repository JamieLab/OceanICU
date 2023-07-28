#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 06/2023

Script to allow plot updates
"""
import fluxengine_driver as fl
#fl.plot_annual_flux('temporal_flux.png',['D:/ESA_CONTRACT/NN/xCO2_SST','D:/ESA_CONTRACT/NN/xCO2_SST_SSS','D:/ESA_CONTRACT/NN/xCO2_SST_SSS_MLD','D:/ESA_CONTRACT/NN/GCB_2023_Watson'],['xCO2_atm+CCI_SST','xCO2_atm+CCI_SST+CMEMS_SSS','xCO2_atm+CCI_SST+CMEMS_SSS+CMEMS_MLD','GCB_Watson_2023'])
fl.plot_annual_flux('GCB_submissions.png',['D:/ESA_CONTRACT/NN/GCB_2023_Watson_CCI','D:/ESA_CONTRACT/NN/GCB_2023_Watson_OISST','D:/ESA_CONTRACT/NN/GCB_2023_Watson_HAD','D:/ESA_CONTRACT/NN/GCB_2023_Watson_HAD_EMLD','D:/ESA_CONTRACT/NN/GCB_2023_Watson_HAD_EMLD_ESSS','D:/ESA_CONTRACT/NN/GCB_2023_Watson_HAD_EMLD_ESSS_RICHFCO2','D:/ESA_CONTRACT/NN/Watson_Recalc'],
  ['GCB_Watson_2023_CCI_SSTv2','GCB_Watson_2023_OISSTv2.1','GCB_Watson_2023_OISSTv2.1_HADSST','GCB_Watson_2023_OISSTv2.1_HADSST_EMLD','GCB_Watson_2023_OISSTv2.1_HADSST_EMLD_ESSS','GCB_Watson_2023_OISSTv2.1_HADSST_EMLD_ESSS_RICHFCO2','GCB2022_Recalculation'],gcb2022=True)

fl.plot_annual_flux('GCB_submissions_ens.png',['D:/ESA_CONTRACT/NN/GCB_2023_Watson_CCI','D:/ESA_CONTRACT/NN/GCB_2023_Watson_Ens1','D:/ESA_CONTRACT/NN/GCB_2023_Watson_Ens2','D:/ESA_CONTRACT/NN/GCB_2023_Watson_Ens3','D:/ESA_CONTRACT/NN/GCB_2023_Watson_Ens4'],
  ['GCB_Watson_2023_Ens1','GCB_Watson_2023_Ens2','GCB_Watson_2023_Ens3','GCB_Watson_2023_Ens4','GCB_Watson_2023_Ens5'],gcb2022=True)

fl.plot_annual_flux('GCB_submissions_ens2.png',['D:/ESA_CONTRACT/NN/GCB_2023_Prelim','D:/ESA_CONTRACT/NN/GCB_2023_Prelim_Ens1','D:/ESA_CONTRACT/NN/GCB_2023_Prelim_Ens2','D:/ESA_CONTRACT/NN/GCB_2023_Prelim_Ens3','D:/ESA_CONTRACT/NN/GCB_2023_Prelim_Ens4'],
  ['GCB_Watson_2023_Ens1','GCB_Watson_2023_Ens2','GCB_Watson_2023_Ens3','GCB_Watson_2023_Ens4','GCB_Watson_2023_Ens5'],gcb2022=True)

fl.plot_annual_flux('GCB_submissions_paired.png',['D:/ESA_CONTRACT/NN/GCB_2023_Watson_CCI','D:/ESA_CONTRACT/NN/GCB_2023_Watson_OISST'],
  ['GCB_Watson_2023_CCI_SSTv2','GCB_Watson_2023_OISSTv2.1'],gcb2022=False)

fl.plot_annual_flux('testing.png',['D:/ESA_CONTRACT/NN/GCB_2023_Prelim_Ens1','D:/ESA_CONTRACT/NN/NEW_TESTING'],
  ['GCB_Watson_2023_CCI_SSTv2','Testing'],gcb2022=False)
