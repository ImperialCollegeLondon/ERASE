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
from kneed import KneeLocator # To compute the knee/elbow of a curve
from generateTrueSensorResponse import generateTrueSensorResponse
from simulateSensorArray import simulateSensorArray
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
import auxiliaryFunctions

os.chdir("..")

# Save flag for figure
saveFlag = False

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15]);

# Number of Adsorbents
numberOfAdsorbents = 20

# Number of Gases
numberOfGases = 2

# Mole Fraction of interest
moleFrac = [0.5, 0.5]

# Sensor ID
sensorID = np.array([17, 15])

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Simulate the sensor response for all possible concentrations
if numberOfGases == 2:
    moleFractionRange = np.array([np.linspace(0,1,10001), 1 - np.linspace(0,1,10001)]).T
elif numberOfGases == 3:
    moleFractionRangeTemp = np.zeros([10001,3])
    num1 = np.random.uniform(0.0,1.0,10001)
    num2 = np.random.uniform(0.0,1.0,10001)
    num3 = np.random.uniform(0.0,1.0,10001)
    sumNum = num1 + num2 + num3
    moleFractionRangeTemp[:,0] = num1/sumNum
    moleFractionRangeTemp[:,1] = num2/sumNum
    moleFractionRangeTemp[:,2] = num3/sumNum
    moleFractionRange = moleFractionRangeTemp[moleFractionRangeTemp[:,0].argsort()]

arraySimResponse = np.zeros([moleFractionRange.shape[0],numberOfGases])
for ii in range(moleFractionRange.shape[0]):
    arraySimResponse[ii,:] = simulateSensorArray(sensorID, pressureTotal, 
                                               temperature, np.array([moleFractionRange[ii,:]]))

# Get the individual sensor reponse for all the given "experimental/test" concentrations
sensorTrueResponse = generateTrueSensorResponse(numberOfAdsorbents,numberOfGases,
                                            pressureTotal,temperature,moleFraction = moleFrac)
# Parse out the true sensor response for the desired sensors in the array
arrayTrueResponse = np.zeros(sensorID.shape[0])
for ii in range(sensorID.shape[0]):
    arrayTrueResponse[ii] = sensorTrueResponse[sensorID[ii]]
arrayTrueResponse = np.tile(arrayTrueResponse,(moleFractionRange.shape[0],1))

# Compute the objective function over all the mole fractions
objFunction = np.sum(np.power((arrayTrueResponse - arraySimResponse)/arrayTrueResponse,2),1)

# Compute the first derivative and the elbow point of sensor 1
firstDerivative = np.zeros([moleFractionRange.shape[0],numberOfGases])
firstDerivative[:,0] = np.gradient(arraySimResponse[:,0])
if all(i >= 0. for i in firstDerivative[:,0]):
    slopeDir1 = "increasing"
else:
    slopeDir1 = "decreasing"
kneedle = KneeLocator(moleFractionRange[:,0], arraySimResponse[:,0], 
                          curve="concave", direction=slopeDir1)
elbowPointS1 = list(kneedle.all_elbows)

# Compute the first derivative and the elbow point of sensor 2
firstDerivative[:,1] = np.gradient(arraySimResponse[:,1])
if all(i >= 0. for i in firstDerivative[:,1]):
    slopeDir2 = "increasing"
else:
    slopeDir2 = "decreasing"
kneedle = KneeLocator(moleFractionRange[:,0], arraySimResponse[:,1], 
                          curve="concave", direction=slopeDir2)
elbowPointS2 = list(kneedle.all_elbows)

# Plot the sensor response for all the conocentrations and highlight the 
# working region
# Obtain coordinates to fill working region
if slopeDir1 == "increasing":
    xFill1 = [0,elbowPointS1[0]]
else:
    xFill1 = [elbowPointS1[0], 1.0]
    
if slopeDir2 == "increasing":
    xFill2 = [0,elbowPointS2[0]]
else:
    xFill2 = [elbowPointS2[0], 1.0]

fig = plt.figure
ax = plt.gca()
# Sensor 1
ax.plot(moleFractionRange[:,0],arraySimResponse[:,0],'r', label = '$s_1$') # Simulated Response
ax.axvline(x=elbowPointS1, linewidth=1, linestyle='dotted', color = 'r') # Elbow point
ax.fill_between(xFill1,1.1*np.max(arraySimResponse), facecolor='red', alpha=0.15)
# Sensor 2
ax.plot(moleFractionRange[:,0],arraySimResponse[:,1],'b', label = '$s_2$') # Simulated Response
ax.axvline(x=elbowPointS2, linewidth=1, linestyle='dotted', color = 'b')  # Elbow point
ax.fill_between(xFill2,1.1*np.max(arraySimResponse), facecolor='blue', alpha=0.15)
# Sensor 3
if numberOfGases == 3:
    ax.axhline(y=arrayTrueResponse[0,2], linewidth=1, linestyle='dotted', 
           color = 'b', label = '$s_2$')
    ax.plot(moleFractionRange[:,0],arraySimResponse[:,2],'g')

ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
ax.set(xlabel='$y_1$ [-]', 
        ylabel='$m_i$ [g kg$^{-1}$]',
        xlim = [0,1], ylim = [0, 1.1*np.max(arraySimResponse)])
ax.legend()

#  Save the figure
if saveFlag:
    # FileName: PureIsotherm_<sensorID>_<currentDateTime>_<GitCommitID_Data>_<GitCommitID_Current>
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
fig = plt.figure
ax = plt.gca()
ax.plot(moleFractionRange[:,0],np.power((arrayTrueResponse[:,0]
                                         -arraySimResponse[:,0])/arrayTrueResponse[:,0],2),'r', label = '$J_1$') # Error first sensor
ax.plot(moleFractionRange[:,0],np.power((arrayTrueResponse[:,1]
                                         -arraySimResponse[:,1])/arrayTrueResponse[:,1],2),'b', label = '$J_2$')  # Error second sensor
if numberOfGases == 3:
    ax.plot(moleFractionRange[:,0],np.power((arrayTrueResponse[:,2]
                                         -arraySimResponse[:,2])/arrayTrueResponse[:,2],2),'g', label = '$J_3$')  # Error third sensor
ax.plot(moleFractionRange[:,0],objFunction,'k', label = '$\Sigma J_i$')  # Error all sensors
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
ax.set(xlabel='$y_1$ [-]', 
        ylabel='$J$ [-]',
        xlim = [0,1.], ylim = [0, None])
ax.legend()

#  Save the figure
if saveFlag:
    # FileName: PureIsotherm_<sensorID>_<currentDateTime>_<GitCommitID_Data>_<GitCommitID_Current>
    sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-')
    moleFrac = str(moleFrac).replace('[','').replace(']','').replace(' ','').replace(',','-').replace('.','')
    saveFileName = "SensorObjFunc_" + sensorText + "_" + moleFrac + "_" + currentDT + "_" + gitCommitID + saveFileExtension
    savePath = os.path.join('simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)

plt.show()