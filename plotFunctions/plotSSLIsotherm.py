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
# Plots the single site Langmuir isotherms
#
# Last modified:
# - 2020-10-28, AK: Add auxiliary functions as a module
# - 2020-10-27, AK: Further improvements and cosmetic changes
# - 2020-10-26, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
import os
import auxiliaryFunctions
from numpy import load
from simulateSSL import simulateSSL
import matplotlib.pyplot as plt
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file

# Flag for saving figure
saveFlag = True

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Sensor ID to be plotted
sensorID = 50

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5])

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15])

# Molefraction
moleFraction = np.array([np.linspace(0,1,101)])

# For now load a given adsorbent isotherm material file
# loadFileName = "isothermParameters_20201020_1756_5f263af.npz" # Two gases
loadFileName = "isothermParameters_20201022_1056_782efa3.npz" # Three gases
hypoAdsorbentFile = os.path.join('..','inputResources',loadFileName);

# Check if the file with the adsorbent properties exist 
if os.path.exists(hypoAdsorbentFile):
    loadedFileContent = load(hypoAdsorbentFile)
    adsorbentIsothermTemp = loadedFileContent['adsIsotherm']
    adsorbentDensityTemp = loadedFileContent['adsDensity']
    molecularWeight = loadedFileContent['molWeight']
else:
    errorString = "Adsorbent property file " + hypoAdsorbentFile + " does not exist."
    raise Exception(errorString)
    
###### TO DO: SERIOUS ISSUE WITH THE ISOTHERM PLOTTING
# Evaluate the isotherms
adsorbentID = np.array([sensorID]) # Do this for consistency
adsorbentIsotherm = adsorbentIsothermTemp[:,:,adsorbentID]
adsorbentDensity = adsorbentDensityTemp[adsorbentID]
equilibriumLoadings = np.zeros([moleFraction.shape[1],adsorbentIsotherm.shape[0]])
# Loop through all the gases so that the single component isotherm is 
# generated. If not multicomponent genretaed. Additionally, several 
# transpose operations are performed to be self-consistent with other codes
for ii in range(adsorbentIsotherm.shape[0]):
    equilibriumLoadings[:,ii] = np.squeeze(simulateSSL(adsorbentIsotherm[ii,:,:].T,adsorbentDensity,
                                      pressureTotal,temperature,moleFraction.T))/adsorbentDensity # [mol/m3]

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Git commit id of the loaded isotherm file
gitCommmitID_loadedFile = hypoAdsorbentFile[-11:-4]

# Plot the pure single component isotherm for the n gases
fig = plt.figure
ax = plt.gca()
# HARD CODED for 3 gases
ax.plot(pressureTotal*moleFraction.T/1.e5, equilibriumLoadings[:,0],
         linewidth=1.5,color='r', label = '$g_1$')
ax.plot(pressureTotal*moleFraction.T/1.e5, equilibriumLoadings[:,1],
         linewidth=1.5,color='b', label = '$g_2$')
ax.plot(pressureTotal*moleFraction.T/1.e5, equilibriumLoadings[:,2],
         linewidth=1.5,color='g', label = '$g_3$')
ax.set(xlabel='$P$ [bar]', 
       ylabel='$q^*$ [mol kg$^{\mathregular{-1}}$]',
       xlim = [0, 1], ylim = [0, 10])
ax.legend()

#  Save the figure
if saveFlag:
    # FileName: PureIsotherm_<sensorID>_<currentDateTime>_<GitCommitID_Data>_<GitCommitID_Current>
    saveFileName = "PureIsotherm_" + str(sensorID) + "_" + currentDT + "_" + gitCommmitID_loadedFile + "_" + gitCommitID + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)
    
# For the figure to be saved show should appear after the save
plt.show()