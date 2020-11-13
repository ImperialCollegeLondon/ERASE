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
import matplotlib.pyplot as plt
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
import auxiliaryFunctions

os.chdir("..")

# Save flag for figure
saveFlag = False

# Plot flag to show standard deviation of errors
plotStdError = True
plotRaw = True
plotBH = False
plotNoise = True

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
moleFrac = [0.1, 0.9]

# Multiplier Error
multiplierError = [1., 1.]

# Sensor ID
sensorID = np.array([6, 2])

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
ax.plot(moleFractionRange[:,0],arraySimResponse[:,0],color ='#1DBDE6', label = '$s_1$') # Simulated Response
ax.fill_between(xFill1,1.1*np.max(arraySimResponse), facecolor='#1DBDE6', alpha=0.25)
# Sensor 2
ax.plot(moleFractionRange[:,0],arraySimResponse[:,1], color = '#F1515E', label = '$s_2$') # Simulated Response
ax.fill_between(xFill2,1.1*np.max(arraySimResponse), facecolor='#F1515E', alpha=0.25)
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
fig = plt.figure
ax = plt.gca()
ax.plot(moleFractionRange[:,0],np.power((arrayTrueResponse[:,0]
                                         -arraySimResponse[:,0])/arrayTrueResponse[:,0],2), color = '#1DBDE6', label = '$J_1$') # Error first sensor
ax.plot(moleFractionRange[:,0],np.power((arrayTrueResponse[:,1]
                                         -arraySimResponse[:,1])/arrayTrueResponse[:,1],2), color = '#F1515E', label = '$J_2$')  # Error second sensor
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

# Plot the objective function used to evaluate the concentration for individual
# sensors and the total (sum)
if plotStdError:
    if set(sensorID) == set([17,15]):
        loadedFileRaw = load("simulationResults/sensitivityAnalysis_17-15_20201109_1033_f7e470f.npz")
        loadedFileBH = load("simulationResults/sensitivityAnalysis_17-15_20201109_1616_63f3499.npz")
        loadedFileNoise = load("simulationResults/sensitivityAnalysis_17-15_20201110_1458_31e3947.npz")
    elif set(sensorID) == set([6,2]):
        loadedFileRaw = load("simulationResults/sensitivityAnalysis_6-2_20201109_1208_f7e470f.npz")
        loadedFileBH = load("simulationResults/sensitivityAnalysis_6-2_20201110_0936_63f3499.npz")
        loadedFileNoise = load("simulationResults/sensitivityAnalysis_6-2_20201110_1458_31e3947.npz")
    elif set(sensorID) == set([17,16]):
        loadedFileRaw = load("simulationResults/sensitivityAnalysis_17-16_20201109_1416_63f3499.npz")
        loadedFileBH = load("simulationResults/sensitivityAnalysis_17-16_20201109_1938_63f3499.npz")
        loadedFileNoise = load("simulationResults/sensitivityAnalysis_17-16_20201110_1458_31e3947.npz")
    elif set(sensorID) == set([17,6]):
        loadedFileRaw = load("simulationResults/sensitivityAnalysis_17-6_20201109_1651_63f3499.npz")
        loadedFileBH = load("simulationResults/sensitivityAnalysis_17-6_20201110_1205_63f3499.npz")
        loadedFileNoise = load("simulationResults/sensitivityAnalysis_17-6_20201110_1458_31e3947.npz")

    # Parse raw data (no noise, default basin hopping iterations (50))
    if plotRaw:
        moleFractionG1_Raw = loadedFileRaw["moleFractionG1"]
        meanConcEstimate_Raw = loadedFileRaw["meanConcEstimate"]
        stdConcEstimate_Raw = loadedFileRaw["stdConcEstimate"]

    # Parse data with higher number of iterations for BH (250)
    if plotBH:
        moleFractionG1_BH = loadedFileBH["moleFractionG1"]
        meanConcEstimate_BH = loadedFileBH["meanConcEstimate"]
        stdConcEstimate_BH = loadedFileBH["stdConcEstimate"]

    # Parse data with noise and default iterations for BH (50)
    if plotNoise:
        moleFractionG1_Noise = loadedFileNoise["moleFractionG1"]
        meanConcEstimate_Noise = loadedFileNoise["meanConcEstimate"]
        stdConcEstimate_Noise = loadedFileNoise["stdConcEstimate"]

    os.chdir("plotFunctions")
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    os.chdir("..")
    
    fig = plt.figure
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    
    ax1.semilogy(np.linspace(0,1,100), np.linspace(0,1,100), 
                 linewidth = 1, linestyle = '--', color = '#adb5bd')
    if plotRaw:
        ax1.semilogy(moleFractionG1_Raw,meanConcEstimate_Raw[:,0], marker='o', 
                     linestyle='None', color='#4f772d', label = 'Reference') # Raw
        ax2.semilogy(moleFractionG1_Raw,stdConcEstimate_Raw[:,0], marker='o', 
                    linewidth = 1, linestyle = ':', color='#4f772d', label = 'Reference') # Raw
    if plotBH:
        ax1.semilogy(moleFractionG1_BH,meanConcEstimate_BH[:,0], marker='o', 
                     linestyle='None', color='#90a955', label = 'More Iterations') # Basin hopping
        ax2.semilogy(moleFractionG1_BH,stdConcEstimate_BH[:,0], marker='o', 
                    linewidth = 1, linestyle = ':', color='#90a955', label = 'More Iterations') # Basin hopping
    if plotNoise:
        ax1.semilogy(moleFractionG1_Noise,meanConcEstimate_Noise[:,0], marker='o', 
                     linestyle='None', color='#90a955', label = 'With Noise') # Noise
        ax2.semilogy(moleFractionG1_Noise,stdConcEstimate_Noise[:,0], marker='o', 
                    linewidth = 1, linestyle = ':', color='#90a955', label = 'With Noise') # Noise
    
    ax2.fill_between(xFill1,1.1*np.max(arraySimResponse), facecolor='#1DBDE6', alpha=0.25)
    ax2.fill_between(xFill2,1.1*np.max(arraySimResponse), facecolor='#F1515E', alpha=0.25)
    
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y")
    ax1.set(xlabel='True $y_1$ [-]', 
            ylabel='Estimated $y_1$ [-]',
            xlim = [0,1.], ylim = [1e-4, 1.])
    
    ax2.locator_params(axis="x", nbins=4)
    ax2.locator_params(axis="y")
    ax2.set(xlabel='$y_1$ [-]', 
            ylabel='$\sigma ({y_1})$ [-]',
            xlim = [0,1.], ylim = [1e-10, 1.])
    ax2.legend()
    
    #  Save the figure
    if saveFlag:
        # FileName: SensorObjFunc_<sensorID>_<noleFrac>_<currentDateTime>_<GitCommitID_Data>>
        sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-')
        saveFileName = "SensorSenAnalStd_" + sensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
    
    plt.show()