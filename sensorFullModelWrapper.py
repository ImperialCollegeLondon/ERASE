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
import matplotlib.pyplot as plt

# Save settings
saveFlag = False
saveFileExtension = ".png"
saveText = "rateConstant"
colorsForPlot = ["1DBDE6","52A2C4","6D95B3","A27A91","CA6678","F1515E"]

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Define the variable to be looped
# This has to be a be a tuple. For on condition write the values followed by a 
# comma to say its a tuple
loopVariable = ([0.001,0.001,0.001],)
    
# Define a dictionary
outputStruct = {}

# Loop over all rate constants
for ii in tqdm(range(len(loopVariable))):
    # Call the full model with a given rate constant
    timeSim, _ , sensorFingerPrint, inputParameters = simulateFullModel(rateConstant = loopVariable[ii])
    outputStruct[ii] = {'timeSim':timeSim,
                'sensorFingerPrint':sensorFingerPrint,
                'inputParameters':inputParameters}
    
# Plot the sensor finger print
os.chdir("plotFunctions")
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax = plt.subplot(1,1,1)
for ii in range(len(loopVariable)):
    timeTemp = outputStruct[ii]["timeSim"]
    fingerPrintTemp = outputStruct[ii]["sensorFingerPrint"]
    ax.plot(timeTemp, fingerPrintTemp,
             linewidth=1.5,color="#"+colorsForPlot[ii],
             label = str(loopVariable[ii][0]))
ax.set(xlabel='$t$ [s]', 
   ylabel='$m_i$ [g kg$^{\mathregular{-1}}$]',
   xlim = [timeSim[0], timeSim[-1]], ylim = [0, 150])
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
#  Save the figure
if saveFlag:
    # FileName: fullModelWrapper_<saveText>_<currentDateTime>_<gitCommitID>
    saveFileName = "fullModelWrapper_" + saveText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName.replace('[','').replace(']',''))
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)
plt.show()
os.chdir("..")