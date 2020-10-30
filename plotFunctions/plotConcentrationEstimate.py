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
# - 2020-10-30, AK: Add zoomed in version
# - 2020-10-29, AK: Improvements to the plots
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
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file

# Flag for saving figure
saveFlag = False

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Plot zoomed plot
plotZoomDist = False

# Gas concentration
molFracG1 = 0.05
molFracG2 = 0.15
molFracG3 = 1 - molFracG1 - molFracG2

# Xlimits and Ylimits
xLimits = [0,1]
yLimits = [0,6] 
xLimitsSum = [0,3]
yLimitsSum = [0,30]

xLimitsZ = [0,1] # Limits for zoom distribution
yLimitsZ = [0,15] # Limits for zoom distribution
xLimitsSumZ = [0,2] # Limits for zoom gas sum
yLimitsSumZ = [0,15] # Limits for zoom gas sum

# Histogram properties
nBins = 100
rangeX = (xLimits)
rangeXS = (xLimitsSum)
histTypeX = 'stepfilled'
alphaX=0.5
densityX = False

# For now load a given adsorbent isotherm material file
# loadFileName = "arrayConcentration_20201030_1109_5c77a62.npy" # One sensor
# loadFileName = "arrayConcentration_20201030_0913_5c77a62.npy" # Two sensors
loadFileName = "arrayConcentration_20201029_2328_5c77a62.npy" # Three sensors
simResultsFile = os.path.join('..','simulationResults',loadFileName);

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Git commit id of the loaded isotherm file
simID_loadedFile = loadFileName[-21:-4]

# Check if the file with the adsorbent properties exist 
if os.path.exists(simResultsFile):
    resultOutput = load(simResultsFile)
    if resultOutput.shape[1] == 4:
        resultOutput = np.delete(resultOutput,0,1)
    elif resultOutput.shape[1] == 5:
        resultOutput = np.delete(resultOutput,[0,1],1)
    elif resultOutput.shape[1] == 6:
        resultOutput = np.delete(resultOutput,[0,1,2],1)
else:
    errorString = "Simulation result file " + simResultsFile + " does not exist."
    raise Exception(errorString)

# Plot the pure single component isotherm for the n gases
fig = plt.figure
# Histogram for gas 1
ax1 = plt.subplot(1,1,1)
ax1.hist(resultOutput[:,0], bins = nBins, range = rangeX, density = densityX, 
         linewidth=1.5, histtype = histTypeX, color='r', alpha = alphaX, label = '$g_1$')
ax1.axvline(x=molFracG1, linewidth=1, linestyle='dotted', color = 'r', alpha = 0.6)

# Histogram for gas 2
# ax2 = plt.subplot(1,4,2)
ax1.hist(resultOutput[:,1], bins = nBins, range = rangeX, density = densityX, 
         linewidth=1.5, histtype = histTypeX, color='b', alpha = alphaX, label = '$g_2$')
ax1.axvline(x=molFracG2, linewidth=1, linestyle='dotted', color = 'b', alpha = 0.6)

# Histogram for gas 3
# ax3 = plt.subplot(1,4,3)
ax1.hist(resultOutput[:,2], bins = nBins, range = rangeX, density = densityX, 
         linewidth=1.5, histtype = histTypeX, color='g', alpha = alphaX, label = '$g_3$')
ax1.axvline(x=molFracG3, linewidth=1, linestyle='dotted', color = 'g', alpha = 0.6)

ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)
ax1.set(xlabel='$y_i$ [-]', 
        ylabel='$f$ [-]',
        xlim = xLimits, ylim = yLimits)
ax1.legend()

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

# Histogram for the sum of mole fraction
fig = plt.figure
ax2 = plt.subplot(1,1,1)
ax2.hist(np.sum(resultOutput,1), bins = nBins, range = rangeXS, density = densityX, 
         linewidth=1.5, histtype = histTypeX, color='k',  alpha = alphaX)
ax2.axvline(x=1., linewidth=1, linestyle='dotted', color = 'k')
ax2.set(xlabel='$\Sigma y_i$ [-]', 
        xlim = xLimitsSum, ylim = yLimitsSum)
ax2.locator_params(axis="x", nbins=4)
ax2.locator_params(axis="y", nbins=4)

#  Save the figure
if saveFlag:
    # FileName: ConcEstimate_<currentDateTime>_<GitCommitID_Current>_<MMdd_GitCommitID_Data>
    saveFileName = "ConcEstimateSum_" + currentDT + "_" + gitCommitID + "_" + simID_loadedFile + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)
    
# For the figure to be saved show should appear after the save
plt.show()

if plotZoomDist:
    # Plot the pure single component isotherm for the n gases
    fig = plt.figure
    # Histogram for gas 1
    ax1 = plt.subplot(1,1,1)
    ax1.hist(resultOutput[:,0], bins = nBins, range = rangeX, density = densityX, 
             linewidth=1.5, histtype = histTypeX, color='r', alpha = alphaX, label = '$g_1$')
    ax1.axvline(x=molFracG1, linewidth=1, linestyle='dotted', color = 'r', alpha = 0.6)
    
    # Histogram for gas 2
    # ax2 = plt.subplot(1,4,2)
    ax1.hist(resultOutput[:,1], bins = nBins, range = rangeX, density = densityX, 
             linewidth=1.5, histtype = histTypeX, color='b', alpha = alphaX, label = '$g_2$')
    ax1.axvline(x=molFracG2, linewidth=1, linestyle='dotted', color = 'b', alpha = 0.6)
    
    # Histogram for gas 3
    # ax3 = plt.subplot(1,4,3)
    ax1.hist(resultOutput[:,2], bins = nBins, range = rangeX, density = densityX, 
             linewidth=1.5, histtype = histTypeX, color='g', alpha = alphaX, label = '$g_3$')
    ax1.axvline(x=molFracG3, linewidth=1, linestyle='dotted', color = 'g', alpha = 0.6)
    
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.set(xlabel='$y_i$ [-]', 
            ylabel='$f$ [-]',
            xlim = xLimitsZ, ylim = yLimitsZ)
    ax1.legend()
    
    #  Save the figure
    if saveFlag:
        # FileName: ConcEstimate_<currentDateTime>_<GitCommitID_Current>_<MMdd_GitCommitID_Data>
        saveFileName = "ConcEstimateZoom_" + currentDT + "_" + gitCommitID + "_" + simID_loadedFile + saveFileExtension
        savePath = os.path.join('..','simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
       
        
    # For the figure to be saved show should appear after the save
    plt.show()
    
    # Histogram for the sum of mole fraction
    fig = plt.figure
    ax2 = plt.subplot(1,1,1)
    ax2.hist(np.sum(resultOutput,1), bins = nBins, range = rangeXS, density = densityX, 
             linewidth=1.5, histtype = histTypeX, color='k',  alpha = alphaX)
    ax2.axvline(x=1., linewidth=1, linestyle='dotted', color = 'k')
    ax2.set(xlabel='$\Sigma y_i$ [-]', 
            xlim = xLimitsSumZ, ylim = yLimitsSumZ)
    ax2.locator_params(axis="x", nbins=4)
    ax2.locator_params(axis="y", nbins=4)
    
    #  Save the figure
    if saveFlag:
        # FileName: ConcEstimate_<currentDateTime>_<GitCommitID_Current>_<MMdd_GitCommitID_Data>
        saveFileName = "ConcEstimateSumZoom_" + currentDT + "_" + gitCommitID + "_" + simID_loadedFile + saveFileExtension
        savePath = os.path.join('..','simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
        
    # For the figure to be saved show should appear after the save
    plt.show()