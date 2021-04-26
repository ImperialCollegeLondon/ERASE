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
# Simulates the full ZLC setup. The model calls the simulate ZLC function 
# to simulate the sorption process and the response is fed to the dead 
# volume simulator
#
# Last modified:
# - 2021-04-22, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
############################################################################

from simulateZLC import simulateZLC
from simulateDeadVolume import simulateDeadVolume
import os
from numpy import load

# Call the simulateZLC function to simulate the sorption in a given sorbent
timeZLC, resultMat, _ = simulateZLC()

# Parse out the mole fraction out from ZLC
moleFracZLC = resultMat[0,:]

# Parse out the flow rate out from ZLC [m3/s]
flowRateZLC = resultMat[3,:]*1e6 # Convert to ccs

# File with parameter estimates for the dead volume
simulationDir = '/Users/ash23win/Google Drive/ERASE/simulationResults/'
fileParameter = 'deadVolumeCharacteristics_20210421_1558_49eb234.npz'
modelOutputTemp = load(simulationDir+fileParameter, allow_pickle=True)["modelOutput"]
# Parse out dead volume parameters
x = modelOutputTemp[()]["variable"]
# Call the simulateDeadVolume function to simulate the dead volume of the setup
_ , _ , moleFracOut = simulateDeadVolume(timeInt = timeZLC,
                                        initMoleFrac = moleFracZLC[0],
                                        feedMoleFrac = moleFracZLC,
                                        flowRate = flowRateZLC,
                                        deadVolume_1 = x[0],
                                        deadVolume_2M = x[1],
                                        deadVolume_2D = x[2],
                                        numTanks_1 = int(x[3]),
                                        flowRate_D = x[4])

# Plot the model response
# Linear scale
os.chdir("plotFunctions")
import matplotlib.pyplot as plt
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax1 = plt.subplot(1,2,1)        
ax1.plot(timeZLC,moleFracZLC,linewidth = 2,color='b',label = 'ZLC') # ZLC response
ax1.plot(timeZLC,moleFracOut,linewidth = 2,color='r',label = 'ZLC+DV') # Combined model response
ax1.set(xlabel='$t$ [s]', 
        ylabel='$y_1$ [-]',
        xlim = [0,300], ylim = [0, 1])   
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)      
ax1.legend()

# Log scale
ax2 = plt.subplot(1,2,2)   
ax2.semilogy(timeZLC,moleFracZLC,linewidth = 2,color='b',label = 'ZLC') # ZLC response       
ax2.semilogy(timeZLC,moleFracOut,linewidth = 2,color='r',label = 'ZLC+DV') # Combined model response    
ax2.set(xlabel='$t$ [s]', 
        xlim = [0,300], ylim = [1e-3, 1])         
ax2.locator_params(axis="x", nbins=4)
ax2.legend()
