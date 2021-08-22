############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2020
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#
# Purpose:
# Plots for the comparison of isotherms obtained from different devices and 
# different fits from ZLC
#
# Last modified:
# - 2021-08-20, AK: Change definition of rate constants
# - 2021-07-01, AK: Cosmetic changes
# - 2021-06-15, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
from computeEquilibriumLoading import computeEquilibriumLoading
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import os
from numpy import load
import auxiliaryFunctions
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Save flag and file name extension
saveFlag = False
saveFileExtension = ".png"

# Colors
colorForPlot = ["faa307","d00000","03071e"]
# colorForPlot = ["E5383B","6C757D"]

# Plot text
plotText = 'DSL'

# Universal gas constant
Rg = 8.314

# Total pressure
pressureTotal = np.array([1.e5]);

# Define temperature
temperature = [308.15, 328.15, 348.15]

# AC Isotherm parameters
x_VOL = [4.65e-1, 1.02e-5 , 2.51e4, 6.51, 3.51e-7, 2.57e4, 100] # (Hassan, QC)

# 13X Isotherm parameters (L pellet)
# x_VOL = [2.50, 2.05e-7, 4.29e4, 4.32, 3.06e-7, 3.10e4, 100] # (Hassan, QC)

# BN Isotherm parameters
# x_VOL = [7.01, 2.32e-07, 2.49e4, 0, 0, 0, 100] # (Hassan, QC)

# ZLC Parameter estimates
# Activated Carbon Experiments
# Pressure and temperature and temperature dependence
zlcFileName = ['zlcParameters_20210812_0905_ea32ed7.npz',
                'zlcParameters_20210812_1850_ea32ed7.npz',
                'zlcParameters_20210813_0348_ea32ed7.npz',
                'zlcParameters_20210813_1321_ea32ed7.npz',
                'zlcParameters_20210813_2133_ea32ed7.npz']

# Create the grid for mole fractions
y = np.linspace(0,1.,100)
# Initialize isotherms 
isoLoading_VOL = np.zeros([len(y),len(temperature)])
isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
kineticConstant_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
objectiveFunction = np.zeros([len(zlcFileName)])

# Loop over all the mole fractions
# Volumetric data
for jj in range(len(temperature)):
    for ii in range(len(y)):
        isoLoading_VOL[ii,jj] = computeEquilibriumLoading(isothermModel=x_VOL[0:-1],
                                                          moleFrac = y[ii],
                                                          temperature = temperature[jj])
# Loop over all available ZLC files
for kk in range(len(zlcFileName)):
    # ZLC Data 
    parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
    parameterReference = load(parameterPath)["parameterReference"]
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
    modelNonDim = modelOutputTemp[()]["variable"] 
    adsorbentDensity = load(parameterPath, allow_pickle=True)["adsorbentDensity"]
    # Print names of files used for the parameter estimation (sanity check)
    fileNameList = load(parameterPath, allow_pickle=True)["fileName"]
    print(fileNameList)

    # Get the date time of the parameter estimates
    parameterDateTime = datetime.strptime(zlcFileName[kk][14:27], '%Y%m%d_%H%M')
    # Date time when the kinetic model shifted from 1 parameter to 2 parameter
    parameterSwitchTime = datetime.strptime('20210616_0800', '%Y%m%d_%H%M')
    # Multiply the paremeters by the reference values
    x_ZLC = np.multiply(modelNonDim,parameterReference)
    print(x_ZLC)
    # When 2 parameter model absent
    if parameterDateTime<parameterSwitchTime:
        isothermModel = x_ZLC[0:-1]
    # When 2 parameter model present
    else:
        isothermModel = x_ZLC[0:-2]
        rateConstant_1 = x_ZLC[-2]
        rateConstant_2 = x_ZLC[-1]

    for jj in range(len(temperature)):
        for ii in range(len(y)):
            isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                 moleFrac = y[ii], 
                                                                 temperature = temperature[jj]) # [mol/kg]
            # Partial pressure of the gas
            partialPressure = y[ii]*pressureTotal
            # delta pressure to compute gradient
            delP = 1e-3
            # Mole fraction (up)
            moleFractionUp = (partialPressure + delP)/pressureTotal
            # Compute the loading [mol/m3] @ moleFractionUp
            equilibriumLoadingUp  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                            temperature=temperature[jj],
                                                            moleFrac=moleFractionUp,
                                                            isothermModel=isothermModel) # [mol/kg]
            
            # Compute the gradient (delq*/dc)
            dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[kk,ii,jj])*adsorbentDensity/(delP/(Rg*temperature[jj])) # [-]

            # Rate constant 1 (analogous to micropore resistance)
            k1 = rateConstant_1
            
            # Rate constant 2 (analogous to macropore resistance)
            k2 = rateConstant_2/dqbydc
            
            # Overall rate constant
            # The following conditions are done for purely numerical reasons
            # If pure (analogous) macropore
            if k1<1e-12:
                rateConstant = k2
            # If pure (analogous) micropore
            elif k2<1e-12:
                rateConstant = k1
            # If both resistances are present
            else:
                rateConstant = 1/(1/k1 + 1/k2)
            
            # Rate constant (overall)
            kineticConstant_ZLC[kk,ii,jj] = rateConstant
        
# Plot the isotherms    
fig = plt.figure
ax1 = plt.subplot(1,2,1)        
for jj in range(len(temperature)):
    ax1.plot(y,isoLoading_VOL[:,jj],color='#'+colorForPlot[jj],label=str(temperature[jj])+' K') # Ronny's isotherm
    for kk in range(len(zlcFileName)):
        ax1.plot(y,isoLoading_ZLC[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.2) # ALL

ax1.set(xlabel='$P$ [bar]', 
ylabel='$q^*$ [mol kg$^\mathregular{-1}$]',
xlim = [0,1], ylim = [0, 3]) 
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)
ax1.legend()   

# Plot the objective function
fig = plt.figure
ax2 = plt.subplot(1,2,2)       
for kk in range(len(zlcFileName)):
    ax2.scatter(kk+1,objectiveFunction[kk]) # ALL

ax2.set(xlabel='Iteration [-]', 
ylabel='$J$ [-]',
xlim = [0,len(zlcFileName)]) 
ax2.locator_params(axis="y", nbins=4)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.locator_params(axis="x", nbins=4)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.legend()   

#  Save the figure
if saveFlag:
    # FileName: isothermComparison_<currentDateTime>_<GitCommitID_Current>_<modelFile>
    saveFileName = "isothermComparison_" + currentDT + "_" + gitCommitID + "_" + zlcFileName[-25:-12] + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if simulationFigures directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)         
plt.show()

# Plot the kinetic constant as a function of mole fraction
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax1 = plt.subplot(1,1,1)        
for jj in range(len(temperature)):
    for kk in range(len(zlcFileName)):
        if kk == 0:
            labelText = str(temperature[jj])+' K'
        else:
            labelText = ''
        ax1.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.5,
                 label=labelText) # ALL

ax1.set(xlabel='$P$ [bar]', 
ylabel='$k$ [s$^\mathregular{-1}$]',
xlim = [0,1], ylim = [0, 1]) 
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)
ax1.legend()   