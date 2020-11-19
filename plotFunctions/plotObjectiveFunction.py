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
# Plots the objective function used for concentration estimation
#
# Last modified:
# - 2020-11-19, AK: Multigas plotting capability
# - 2020-11-17, AK: Multisensor plotting capability
# - 2020-11-11, AK: Cosmetic changes and add standard deviation plot
# - 2020-11-05, AK: Initial creation
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
from kneed import KneeLocator # To compute the knee/elbow of a curve
from generateTrueSensorResponse import generateTrueSensorResponse
from simulateSensorArray import simulateSensorArray
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
import auxiliaryFunctions

os.chdir("..")

# Save flag for figure
saveFlag = False

# Plot flag to show standard deviation of errors
plotStdError = False
plotRaw = True
plotBH = False
plotNoise = True

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Plotting colors
colorsForPlot = ["ff499e","d264b6","a480cf","779be7","49b6ff"]

# Number of molefractions
numMolFrac= 1001

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15]);

# Number of Adsorbents
numberOfAdsorbents = 20

# Number of Gases
numberOfGases = 3

# Third gas mole fraction
thirdGasMoleFrac = 0.25

# Mole Fraction of interest
moleFrac = [0.1, 0.9]

# Multiplier Error
multiplierError = [1., 1., 1.]

# Sensor ID
sensorID = np.array([5,6,8])

# Acceptable SNR
signalToNoise = 25*0.1

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Simulate the sensor response for all possible concentrations
if numberOfGases == 2:
    moleFractionRange = np.array([np.linspace(0,1,numMolFrac), 1 - np.linspace(0,1,numMolFrac)]).T
elif numberOfGases == 3:
    remainingMoleFrac = 1. - thirdGasMoleFrac
    moleFractionRange = np.array([np.linspace(0,remainingMoleFrac,numMolFrac), 
                                  remainingMoleFrac - np.linspace(0,remainingMoleFrac,numMolFrac),
                                  np.tile(thirdGasMoleFrac,numMolFrac)]).T

arraySimResponse = np.zeros([moleFractionRange.shape[0],sensorID.shape[0]])
for ii in range(moleFractionRange.shape[0]):
    arraySimResponse[ii,:] = simulateSensorArray(sensorID, pressureTotal, 
                                               temperature, np.array([moleFractionRange[ii,:]])) * multiplierError

# Get the individual sensor reponse for all the given "experimental/test" concentrations
sensorTrueResponse = generateTrueSensorResponse(numberOfAdsorbents,numberOfGases,
                                            pressureTotal,temperature,moleFraction = moleFrac)
# Parse out the true sensor response for the desired sensors in the array
arrayTrueResponse = np.zeros(sensorID.shape[0])
for ii in range(sensorID.shape[0]):
    arrayTrueResponse[ii] = sensorTrueResponse[sensorID[ii]]*multiplierError[ii]
arrayTrueResponse = np.tile(arrayTrueResponse,(moleFractionRange.shape[0],1))

# Compute the objective function over all the mole fractions
objFunction = np.sum(np.power((arrayTrueResponse - arraySimResponse)/arrayTrueResponse,2),1)

# Compute the first derivative, elbow point, and the fill regions for all
# sensors for 2 gases

xFill = np.zeros([arraySimResponse.shape[1],2])
# Loop through all sensors
for kk in range(arraySimResponse.shape[1]):
    firstDerivative = np.zeros([arraySimResponse.shape[0],1])
    firstDerivative[:,0] = np.gradient(arraySimResponse[:,kk])
    if all(i >= 0. for i in firstDerivative[:,0]):
        slopeDir = "increasing"
    else:
        slopeDir = "decreasing"
    kneedle = KneeLocator(moleFractionRange[:,0], arraySimResponse[:,kk], 
                          curve="concave", direction=slopeDir)
    elbowPoint = list(kneedle.all_elbows)
    
    # Plot the sensor response for all the conocentrations and highlight the 
    # working region
    # Obtain coordinates to fill working region
    if slopeDir == "increasing":
        xFill[kk,:] = [0,elbowPoint[0]]
    else:
        if numberOfGases == 2:
            xFill[kk,:] = [elbowPoint[0], 1.0]
        elif numberOfGases == 3:
            xFill[kk,:] = [elbowPoint[0], 1.-thirdGasMoleFrac]

fig = plt.figure
ax = plt.gca()
# Loop through all sensors
for kk in range(arraySimResponse.shape[1]):
    ax.plot(moleFractionRange[:,0],arraySimResponse[:,kk],color='#'+colorsForPlot[kk], label = '$s_'+str(kk+1)+'$') # Simulated Response
    ax.fill_between(xFill[kk,:],1.1*np.max(arraySimResponse), facecolor='#'+colorsForPlot[kk], alpha=0.25)
    if numberOfGases == 2:
        ax.fill_between([0.,1.],signalToNoise, facecolor='#4a5759', alpha=0.25) 
    elif numberOfGases == 3:
        mpl.rcParams['hatch.linewidth'] = 0.1 
        ax.fill_between([0.,1.-thirdGasMoleFrac],signalToNoise, facecolor='#4a5759', alpha=0.25)
        ax.fill_between([1.-thirdGasMoleFrac,1.],1.1*np.max(arraySimResponse), facecolor='#555b6e', alpha=0.25, hatch = 'x') 
ax.set(xlabel='$y_1$ [-]', 
       ylabel='$m_i$ [g kg$^{-1}$]',
       xlim = [0,1], ylim = [0, 1.1*np.max(arraySimResponse)])     
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
ax.legend()

#  Save the figure
if saveFlag:
    # FileName: SensorResponse_<sensorID>_<currentDateTime>_<GitCommitID_Current>
    sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-')
    saveFileName = "SensorResponse_" + sensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
    savePath = os.path.join('simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)
plt.show()

# Plot the objective function used to evaluate the concentration for individual
# sensors and the total (sum)
if numberOfGases == 2:
    fig = plt.figure
    ax = plt.gca()
    for kk in range(arraySimResponse.shape[1]): 
        ax.plot(moleFractionRange[:,0],np.power((arrayTrueResponse[:,kk]
                                                  -arraySimResponse[:,kk])/arrayTrueResponse[:,kk],2),
                color='#'+colorsForPlot[kk], label = '$J_'+str(kk+1)+'$') 
    ax.plot(moleFractionRange[:,0],objFunction,color='#'+colorsForPlot[-1], label = '$\Sigma J_i$')  # Error all sensors
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    ax.set(xlabel='$y_1$ [-]', 
            ylabel='$J$ [-]',
            xlim = [0,1.], ylim = [0, None])
    ax.legend()
    
    #  Save the figure
    if saveFlag:
        # FileName: SensorObjFunc_<sensorID>_<noleFrac>_<currentDateTime>_<GitCommitID_Current>
        sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-')
        moleFrac = str(moleFrac).replace('[','').replace(']','').replace(' ','').replace(',','-').replace('.','')
        saveFileName = "SensorObjFunc_" + sensorText + "_" + moleFrac + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('simulationFigures',saveFileName)
        # Check if simulationFigures directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
    plt.show()