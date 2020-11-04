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
#
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
from generateTrueSensorResponse import generateTrueSensorResponse
import matplotlib.pyplot as plt
import auxiliaryFunctions
from scipy import stats
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
import os
os.chdir("..")

# Histogram properties
nBins = 100
rangeX = ([-10,10])
histTypeX = 'step'
alphaX=0.5
densityX = False


# Flag for saving figures
saveFlag = False

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Sensors to plot
sensorID = range(0,20)

# Sensor ID for plotting histogram
histSensorID = 16

# Total pressure of the gas [Pa]
numberOfGases = 2

# Mole fraction to be plotted
moleFraction = np.linspace(0,1,1001)

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5])

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15])

# Total number of sensor elements/gases simulated and generated using 
# generateHypotheticalAdsorbents.py function
numberOfAdsorbents = 20

sensorTrueResponse = generateTrueSensorResponse(numberOfAdsorbents, numberOfGases,
                                                pressureTotal, temperature)

# Parse out sensors that need to be plotted
sensorPlotResponse = np.zeros([len(sensorID),sensorTrueResponse.shape[1]])
for jj in range(len(sensorID)):
    for ii in range(sensorTrueResponse.shape[1]):
        sensorPlotResponse[jj,ii] = (sensorTrueResponse[sensorID[jj],ii]) 

# Perform a standard normal variate of the sensor output    
meanResponse = np.mean(sensorPlotResponse,axis=1)
stdResponse = np.std(sensorPlotResponse,axis=1)

sensorSNVResponse = np.zeros([len(sensorID),sensorTrueResponse.shape[1]])
for jj in range(len(sensorID)):
    sensorSNVResponse[jj,:] = (sensorPlotResponse[jj,:] - meanResponse[jj])/stdResponse[jj]

# Perform a Box-Cpx transformation
lambdaParam = np.zeros(len(sensorID))
for jj in range(len(sensorID)):
    _ , lambdaParam[jj] = stats.boxcox(sensorPlotResponse[jj,:])

# Calculate the skewness of the isotherm
skewnessSNVResponse = np.zeros(len(sensorID))
for jj in range(len(sensorID)):
    skewnessNum = np.sum(np.power(sensorSNVResponse[jj,:] - np.mean(sensorSNVResponse[jj,:]),3))
    skewnessDen = (len(sensorSNVResponse[jj,:]))*np.power(np.std(sensorSNVResponse[jj,:]),3)
    skewnessSNVResponse[jj] = skewnessNum/skewnessDen
    
# Calculate the kurtosis of the isotherm
kurtosisSNVResponse = np.zeros(len(sensorID))
for jj in range(len(sensorID)):
    kurtosisNum = np.sum(np.power(sensorSNVResponse[jj,:] - np.mean(sensorSNVResponse[jj,:]),4))
    kurtosisDen = (len(sensorSNVResponse[jj,:]))*np.power(np.std(sensorSNVResponse[jj,:]),4)
    kurtosisSNVResponse[jj] = kurtosisNum/kurtosisDen

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()
    
# Plot the sensor finger print for different concentrations of gases
fig = plt.figure
ax = plt.subplot(1,2,1)      
for ii in range(len(sensorID)):
    plotX = np.ones(sensorTrueResponse.shape[1])*sensorID[ii]
    plotY = sensorPlotResponse[ii,:]
    s1 = ax.scatter(plotX,plotY,c = moleFraction, cmap='RdYlBu')

ax.set(xlabel='Adsorbent ID [-]',
       ylabel='$m_i$ [g kg$^{-1}$]',
        xlim = [0, 20], ylim = [0,300])
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
plt.colorbar(s1,ax=ax)   
 
#  Save the figure
if saveFlag:
    # FileName: PureIsotherm_<sensorID>_<currentDateTime>_<GitCommitID_Data>_<GitCommitID_Current>
    saveFileName = "SensorMapRaw_" + currentDT + "_" + gitCommitID + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)

# Plot the sensor finger print for different concentrations of gases, but with
# mean centering and normalizing with standard deviation
ax = plt.subplot(1,2,2)      
for ii in range(len(sensorID)):
    plotX = np.ones(sensorTrueResponse.shape[1])*sensorID[ii]
    plotY = sensorSNVResponse[ii,:]
    s2 = ax.scatter(plotX,plotY,c = moleFraction, cmap='RdYlBu')
ax.set(xlabel='Adsorbent ID [-]',
       ylabel='$\hat{m}_i$ [-]',
        xlim = [0, 20], ylim = [-10,5])
ax.locator_params(axis="x", nbins=4)
ax.locator_params(axis="y", nbins=4)
plt.colorbar(s2,ax=ax)    
#  Save the figure
if saveFlag:
    # FileName: PureIsotherm_<sensorID>_<currentDateTime>_<GitCommitID_Data>_<GitCommitID_Current>
    saveFileName = "SensorMapSNV_" + currentDT + "_" + gitCommitID + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)

plt.show()