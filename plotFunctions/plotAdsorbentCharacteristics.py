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
# Plots the adsorbent properties for the hypothetical materials
#
# Last modified:
# - 2020-10-28, AK: Add auxiliary functions as a module
# - 2020-10-27, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import os
from numpy import load
import auxiliaryFunctions
import matplotlib.pyplot as plt
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

# Flag for saving figure
saveFlag = False

# Save file extension (png or pdf)
saveFileExtension = ".pdf"

# Select the id of gas that is to be plotted
gasID = 0

# For now load a given adsorbent isotherm material file
# loadFileName = "isothermParameters_20201020_1756_5f263af.npz" # Two gases
loadFileName = "isothermParameters_20201022_1056_782efa3.npz" # Three gases
hypoAdsorbentFile = os.path.join('..','inputResources',loadFileName);

# Check if the file with the adsorbent properties exist 
if os.path.exists(hypoAdsorbentFile):
    loadedFileContent = load(hypoAdsorbentFile)
    adsorbentIsothermTemp = loadedFileContent['adsIsotherm']
    adsorbentIsotherm = adsorbentIsothermTemp[gasID,:,:]
    adsorbentDensity = loadedFileContent['adsDensity']
    molecularWeight = loadedFileContent['molWeight']
else:
    errorString = "Adsorbent property file " + hypoAdsorbentFile + " does not exist."
    raise Exception(errorString)
    
# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Git commit id of the loaded isotherm file
gitCommmitID_loadedFile = hypoAdsorbentFile[-11:-4]

# Plot the pure single component isotherm for the n gases
colorVar = range(1,101)
fig = plt.figure
ax1 = plt.subplot(1,3,1)
s1 = ax1.scatter(adsorbentIsotherm[0,:], adsorbentIsotherm[1,:], c = colorVar, cmap='RdYlBu')
ax1.set(xlabel='$q_\mathregular{sat}$ [mol kg$^{\mathregular{-1}}$]',
        ylabel='$b_\mathregular{0}$ [m$^{\mathregular{3}}$ mol$^{\mathregular{-1}}$]',
       xlim = [0, 10], ylim = [0, 3e-6])
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)

ax2 = plt.subplot(1,3,2)
s2 = ax2.scatter(adsorbentIsotherm[0,:], -adsorbentIsotherm[2,:], c = colorVar, cmap='RdYlBu')
ax2.set(xlabel='$q_\mathregular{sat}$ [mol kg$^{\mathregular{-1}}$]',
        ylabel='-$\Delta H$ [J mol$^{\mathregular{-1}}$]',
       xlim = [0, 10], ylim = [0, 4e4])
ax2.locator_params(axis="x", nbins=4)
ax2.locator_params(axis="y", nbins=4)

ax3 = plt.subplot(1,3,3)
s3 = ax3.scatter(adsorbentIsotherm[0,:], adsorbentDensity, c = colorVar, cmap='RdYlBu')
ax3.set(xlabel='$q_\mathregular{sat}$ [mol kg$^{\mathregular{-1}}$]',
        ylabel='$\\rho$ [kg m$^{\mathregular{-3}}$]',
       xlim = [0, 10], ylim = [500, 1500])
ax3.locator_params(axis="x", nbins=4)
ax3.locator_params(axis="y", nbins=4)

#  Save the figure
if saveFlag:
    # FileName: PureIsotherm_<sensorID>_<currentDateTime>_<GitCommitID_Data>_<GitCommitID_Current>
    saveFileName = "AdsCharac_" + str(gasID) + "_" + currentDT + "_" + gitCommmitID_loadedFile + "_" + gitCommitID + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)

# For the figure to be saved show should appear after the save
plt.show()