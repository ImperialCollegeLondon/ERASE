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
# Plots to visualize estimated concentration
#
# Last modified:
# - 2020-10-23, AK: Initial creation
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

# Flag for saving figure
saveFlag = False

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Xlimits and Ylimits
xLimits = [0,1]
xLimitsSum = [0,2]
yLimits = [0,40]

# Number of bins for histogram
nBins = 10 

# For now load a given adsorbent isotherm material file
# loadFileName = "arrayConcentration_20201026_1239_6bf5c35.npy" # Three gases
loadFileName = "arrayConcentration_20201026_1240_6bf5c35.npy" # Two sensors
simResultsFile = os.path.join('..','simulationResults',loadFileName);

# Check if the file with the adsorbent properties exist 
if os.path.exists(simResultsFile):
    resultOutput = load(simResultsFile)
    if resultOutput.shape[1] == 4:
        resultOutput = np.delete(resultOutput,0,1)
    elif resultOutput.shape[1] == 5:
        resultOutput = np.delete(resultOutput,[0,1],1)
else:
    errorString = "Simulation result file " + simResultsFile + " does not exist."
    raise Exception(errorString)

# Plot the pure single component isotherm for the n gases
fig = plt.figure
# Histogram for gas 1
ax1 = plt.subplot(1,4,1)
ax1.hist(resultOutput[:,0], bins = nBins,
         linewidth=1.5, histtype = 'step', color='r', label = '$g_1$')
ax1.set(xlabel='$y_1$ [-]', 
        ylabel='$f$ [-]',
        xlim = xLimits, ylim = yLimits)
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)

# Histogram for gas 2
ax2 = plt.subplot(1,4,2)
ax2.hist(resultOutput[:,1], bins = nBins,
         linewidth=1.5, histtype = 'step', color='b', label = '$g_2$')
ax2.set(xlabel='$y_2$ [-]', 
        xlim = xLimits, ylim = yLimits)
ax2.locator_params(axis="x", nbins=4)
ax2.locator_params(axis="y", nbins=4)

# Histogram for gas 3
ax3 = plt.subplot(1,4,3)
ax3.hist(resultOutput[:,2], bins = nBins,
         linewidth=1.5, histtype = 'step', color='g', label = '$g_3$')
ax3.set(xlabel='$y_3$ [-]', 
        xlim = xLimits, ylim = yLimits)
ax3.locator_params(axis="x", nbins=4)
ax3.locator_params(axis="y", nbins=4)

# Histogram for the sum of mole fraction
ax3 = plt.subplot(1,4,4)
ax3.hist(np.sum(resultOutput,1), bins = nBins,
         linewidth=1.5, histtype = 'step', color='k')
ax3.set(xlabel='$\Sigma y$ [-]', 
        xlim = xLimitsSum, ylim = yLimits)
ax3.locator_params(axis="x", nbins=4)
ax3.locator_params(axis="y", nbins=4)

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Git commit id of the loaded isotherm file
simID_loadedFile = loadFileName[-21:-4]

#  Save the figure
if saveFlag:
    # FileName: ConcEstimate_<currentDateTime>_<GitCommitID_Current>_<MMdd_GitCommitID_Data>
    saveFileName = "ConcEstimate_" + currentDT + "_" + gitCommitID + "_" + simID_loadedFile + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)
    
# For the figure to be saved show should appear after the save
plt.show()