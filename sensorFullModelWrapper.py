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
# Wrapper function for sensor full model to check the impact of different 
# variables
#
# Last modified:
# - 2021-03-23, AK: Minor fixes
# - 2021-02-03, AK: Change output plot response to absolute values
# - 2021-01-20, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import auxiliaryFunctions
from simulateFullModel import simulateFullModel
from tqdm import tqdm # To track progress of the loop
import numpy as np
import os
from numpy import savez
import socket
import matplotlib.pyplot as plt

# Save and plot settings
saveFlag = True
saveFileExtension = ".png"
saveFlag = "rateConstant"
colorsForPlot = ["E5383B","CD4448","B55055","9C5D63","846970","6C757D"]

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Define the variable to be looped
# This has to be a be a tuple. For on condition write the values followed by a 
# comma to say its a tuple
volTotal = 10e-7
voidFrac = 0.5
loopVariable = ([0.0001,0.0001,0.0001],
                [0.0005,0.0005,0.0005],
                [0.001,0.001,0.001],
                [0.005,0.005,0.005],
                [0.01,0.01,0.01],
                [10000.0,10000.0,10000.0])
    
# Define a dictionary
outputStruct = {}

# Loop over all the individual elements of the loop variable
for ii in tqdm(range(len(loopVariable))):
    # Call the full model with a given rate constant
    timeSim, _ , sensorFingerPrint, inputParameters = simulateFullModel(volTotal = volTotal,
                                                                        voidFrac = voidFrac,
                                                                        rateConstant = loopVariable[ii])
    outputStruct[ii] = {'timeSim':timeSim,
                'sensorFingerPrint':sensorFingerPrint,
                'inputParameters':inputParameters}
    
# Save the array concentration into a native numpy file
# The .npz file is saved in a folder called simulationResults (hardcoded)
#  Save the figure
if saveFlag:
    filePrefix = "fullModelSensitivity"
    saveFileName = filePrefix + "_" + saveFlag + "_" + currentDT + "_" + gitCommitID;
    savePath = os.path.join('simulationResults',saveFileName)
    savez (savePath, outputStruct = outputStruct, # True response
            hostName = socket.gethostname()) # Hostname of the computer    

# Plot the sensor finger print
os.chdir("plotFunctions")
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax = plt.subplot(1,1,1)
for ii in range(len(loopVariable)):
    timeTemp = outputStruct[ii]["timeSim"]
    fingerPrintTemp = (outputStruct[ii]["sensorFingerPrint"]
                       *outputStruct[ii]["inputParameters"][1]
                       *outputStruct[ii]["inputParameters"][9]) # Compute true response [g]
    ax.plot(timeTemp, fingerPrintTemp,
             linewidth=1.5,color="#"+colorsForPlot[ii],
             label = str(loopVariable[ii]))
ax.set(xlabel='$t$ [s]', 
   ylabel='$m_i$ [g]',
   xlim = [timeSim[0], timeSim[-1]], ylim = [0, None])
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
plt.show()
os.chdir("..")