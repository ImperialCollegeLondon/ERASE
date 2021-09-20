############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2021
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#
# Purpose:
# Plotting function to compare the concentration estimates obtained from using
# the full model and one from assuming instantaneous equilibrium (non kinetics)
#
# Last modified:
# - 2021-01-26, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
from numpy import load
import os
import auxiliaryFunctions
import matplotlib.pyplot as plt
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Save flag
saveFlag = True

# Color combo
colorForMat  = ["ff006e","3a86ff"]
colorForConc  = ["fe7f2d","619b8a","233d4d"]

# Y limits for the plot
X_LIMITS = [0.,1000.]
scaleLog = True

# Legend First
legendText = ["Without Noise", "With Noise"]
legendFlag = False

# File name for equilibrium model and full model estimates
loadFileName_E = "fullModelConcentrationEstimate_6-2_20210126_1810_848451b.npz" # Eqbm
loadFileName_F = "fullModelConcentrationEstimate_6-2_20210126_1434_848451b.npz" # Full model

# Parse out equilbirum results file
simResultsFile = os.path.join('..','simulationResults',loadFileName_E);
loadedFile_E = load(simResultsFile, allow_pickle=True)
concentrationEstimate_E = loadedFile_E["arrayConcentration"]

# Parse out full model results file
simResultsFile = os.path.join('..','simulationResults',loadFileName_F);
loadedFile_F = load(simResultsFile, allow_pickle=True)
concentrationEstimate_F = loadedFile_F["arrayConcentration"]

# Parse out true responses (this should be the same for both eqbm and full
# model (here loaded from eqbm)
trueResponseStruct = loadedFile_F["outputStruct"].item()
# Parse out time
timeSim = []
timeSim = trueResponseStruct[0]["timeSim"]
# Parse out feed mole fraction
feedMoleFrac = trueResponseStruct[0]["inputParameters"][4]
# Parse out true sensor finger print
sensorFingerPrint = np.zeros([len(timeSim),len(trueResponseStruct)])
for ii in range(len(trueResponseStruct)):
    sensorFingerPrint[:,ii] = trueResponseStruct[ii]["sensorFingerPrint"]

# Points that will be taken for sampling (for plots)
lenSampling = 6
fig = plt.figure
# Plot the true sensor response (using the full model)
ax = plt.subplot(1,2,1)
ax.plot(timeSim[0:len(timeSim):lenSampling],
        sensorFingerPrint[0:len(timeSim):lenSampling,0],
        marker = 'o', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
        color='#'+colorForMat[0], label = str(trueResponseStruct[0]["inputParameters"][0]).replace('[','').replace(']',''))
ax.plot(timeSim[0:len(timeSim):lenSampling],
        sensorFingerPrint[0:len(timeSim):lenSampling,1],
        marker = 'o', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
        color='#'+colorForMat[1], label = str(trueResponseStruct[1]["inputParameters"][0]).replace('[','').replace(']',''))
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
ax.set(xlabel='$t$ [s]', 
        ylabel='$m_i$ [g kg$^{\mathregular{-1}}$]',
        xlim = X_LIMITS, ylim = [0, 30])
ax.legend()

# Plot the evolution of the gas composition with respect to time
ax = plt.subplot(1,2,2)
ax.plot([timeSim[0],timeSim[-1]], [feedMoleFrac[0],feedMoleFrac[0]],
        linestyle = 'dashed', linewidth = 1.,
        color='#'+colorForConc[2])
ax.plot(timeSim[0:len(timeSim):lenSampling],
        concentrationEstimate_E[0:len(timeSim):lenSampling,2],
        marker = 'v', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
        color='#'+colorForConc[0], label = 'Equilibrium')
ax.plot(timeSim[0:len(timeSim):lenSampling],
        concentrationEstimate_F[0:len(timeSim):lenSampling,2],
        marker = 'o', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
        color='#'+colorForConc[1], label = 'Full Model')  
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
ax.set(xlabel='$t$ [s]', 
        ylabel='$y_1$ [-]',
        xlim = X_LIMITS, ylim = [0, 0.2])
ax.legend()
plt.show()