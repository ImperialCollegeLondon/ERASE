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
colorForPlot = ["FF1B6B","A273B5","45CAFF"]
# colorForPlot = ["E5383B","6C757D"]

# Plot text
plotText = 'DSL'

# Define temperature
temperature = [306.44, 325.98, 345.17]

# AC Isotherm parameters
x_VOL = [0.44, 3.17e-6, 28.63e3, 6.10, 3.21e-6, 20.37e3, 100] # (Pini 2020)
# x_VOL = [2.81e-5, 1.25e-7, 2.07e2, 4.12, 7.29e-7, 2.65e4, 100] # (Hassan, QC)

# 13X Isotherm parameters
# x_VOL = [2.50, 2.05e-7, 4.29e4, 4.32, 3.06e-7, 3.10e4, 100] # (Hassan, QC)

# ZLC parameter estimate files
# Experiment 43 and 48
# zlcFileName = ['zlcParameters_20210618_1837_36d3aa3.npz',
#                 'zlcParameters_20210618_2209_36d3aa3.npz',
#                 'zlcParameters_20210619_0128_36d3aa3.npz',
#                 'zlcParameters_20210619_0447_36d3aa3.npz',
#                 'zlcParameters_20210619_0759_36d3aa3.npz',]

# Experiment 60 - 
zlcFileName = ['zlcParameters_20210701_0843_4fd9c19.npz',]

# Create the grid for mole fractions
y = np.linspace(0,1.,100)
# Initialize isotherms 
isoLoading_VOL = np.zeros([len(y),len(temperature)])
isoLoading_VOL_HA = np.zeros([len(y),len(temperature)])
isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
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

    # Get the date time of the parameter estimates
    parameterDateTime = datetime.strptime(zlcFileName[kk][14:27], '%Y%m%d_%H%M')
    # Date time when the kinetic model shifted from 1 parameter to 2 parameter
    parameterSwitchTime = datetime.strptime('20210616_0800', '%Y%m%d_%H%M')
    # Multiply the paremeters by the reference values
    x_ZLC = np.multiply(modelNonDim,parameterReference)
    # When 2 parameter model absent
    if parameterDateTime<parameterSwitchTime:
        isothermModel = x_ZLC[0:-1]
    # When 2 parameter model present
    else:
        isothermModel = x_ZLC[0:-2]

    for jj in range(len(temperature)):
        for ii in range(len(y)):
            isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                 moleFrac = y[ii], 
                                                                 temperature = temperature[jj])
        
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