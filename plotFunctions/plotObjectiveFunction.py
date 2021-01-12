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
# - 2021-01-11, AK: Cosmetic changes
# - 2020-11-27, AK: More plotting fix for 3 gas system
# - 2020-11-24, AK: Fix for 3 gas system
# - 2020-11-23, AK: Change ternary plots
# - 2020-11-20, AK: Introduce ternary plots
# - 2020-11-19, AK: Add 3 gas knee calculator
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
import pdb
import numpy as np
from numpy import load
from kneed import KneeLocator # To compute the knee/elbow of a curve
from generateTrueSensorResponse import generateTrueSensorResponse
from simulateSensorArray import simulateSensorArray
import os
from sklearn.cluster import KMeans
import pandas as pd
import ternary
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
import auxiliaryFunctions

os.chdir("..")

# Save flag for figure
saveFlag = False

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Plotting colors
colorsForPlot = ["ff499e","d264b6","a480cf","779be7","49b6ff"]
colorsForPlot = ["eac435","345995","03cea4","fb4d3d","ca1551"]
colorGroup = ["#f94144","#43aa8b"]
colorIntersection = ["ff595e","ffca3a","8ac926","1982c4","6a4c93"]

# Number of molefractions
numMolFrac= 10001

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15]);

# Number of Adsorbents
numberOfAdsorbents = 60

# Number of Gases
numberOfGases = 3

# Experimental mole fraction for 3 gases
if numberOfGases == 3:
    inputMoleFracALL = np.array([[0.01, 0.99, 0.00],
                                 [0.10, 0.90, 0.00],
                                 [0.25, 0.75, 0.00],
                                 [0.50, 0.50, 0.00],
                                 [0.75, 0.25, 0.00],
                                 [0.90, 0.10, 0.00],
                                 [0.99, 0.01, 0.00]])

    # Fix one gas
    fixOneGas = False
    # Third gas mole fraction
    thirdGasMoleFrac = 0.

# Mole Fraction of interest
moleFrac = [0.3, 0.7]

# Multiplier Error
multiplierError = [1, 1.,1.]

# Sensor ID
sensorID = np.array([6,12,55])
saveFileSensorText = [6,12,55]

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
    moleFractionRangeTemp = np.random.dirichlet((1,1,1),numMolFrac)
    moleFractionRange = moleFractionRangeTemp[moleFractionRangeTemp[:,0].argsort()]
    if fixOneGas:
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
if numberOfGases == 2 or (numberOfGases == 3 and fixOneGas == True):
    xFill = np.zeros([arraySimResponse.shape[1],2])
    # Loop through all sensors
    firstDerivative = np.zeros([arraySimResponse.shape[0],arraySimResponse.shape[1]])
    firstDerivativeSimResponse_y1 = np.zeros([moleFractionRange.shape[0],arraySimResponse.shape[1]])
    firstDerivativeSimResponse_y2 = np.zeros([moleFractionRange.shape[0],arraySimResponse.shape[1]])
    secondDerivative = np.zeros([firstDerivative.shape[0],firstDerivative.shape[1]])
    for kk in range(arraySimResponse.shape[1]):
        firstDerivative[:,kk] = np.gradient(arraySimResponse[:,kk],moleFractionRange[1,0]-moleFractionRange[0,0])
        secondDerivative[:,kk] = np.gradient(firstDerivative[:,kk],moleFractionRange[1,0]-moleFractionRange[0,0])
        firstDerivativeSimResponse_y1[:,kk] = np.gradient(moleFractionRange[:,0],arraySimResponse[:,kk])
        firstDerivativeSimResponse_y2[:,kk] = np.gradient(moleFractionRange[:,1],arraySimResponse[:,kk])
        # Get the sign of the first derivative for increasing/decreasing
        if all(i >= 0. for i in firstDerivative[:,kk]):
            slopeDir = "increasing"
        elif all(i < 0. for i in firstDerivative[:,kk]):
            slopeDir = "decreasing"
        else:
            print("Dangerous! I should not be here!!!")
        # Get the sign of the second derivative for concavity/convexity
        if all(i >= 0. for i in secondDerivative[:,kk]):
            secondDerDir = "convex"
        elif all(i < 0. for i in secondDerivative[:,kk]):
            secondDerDir = "concave"
        else:
            print("Dangerous! I should not be here!!!")
    
    
        kneedle = KneeLocator(moleFractionRange[:,0], arraySimResponse[:,kk], 
                              curve=secondDerDir, direction=slopeDir)
        elbowPoint = list(kneedle.all_elbows)

        # Obtain coordinates to fill working region
        if secondDerDir == "concave":
            if slopeDir == "increasing":
                xFill[kk,:] = [0,elbowPoint[0]]
            else:
                if numberOfGases == 2:
                    xFill[kk,:] = [elbowPoint[0], 1.0]
                elif numberOfGases == 3:
                    if fixOneGas:
                        xFill[kk,:] = [elbowPoint[0], 1.-thirdGasMoleFrac]
        elif secondDerDir == "convex":
            if slopeDir == "increasing":
                if numberOfGases == 3:
                    if fixOneGas:
                        xFill[kk,:] = [elbowPoint[0],1.-thirdGasMoleFrac]
            else:
                    xFill[kk,:] = [0,elbowPoint[0]]
        else:
            print("Dangerous! I should not be here!!!")      

# Start plotting
if numberOfGases == 2  or (numberOfGases == 3 and fixOneGas == True):
    if numberOfGases == 2:
        fig = plt.figure
        ax = plt.gca()
    elif numberOfGases == 3:
        fig = plt.figure
        ax = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)        
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
                ax2.plot(moleFractionRange[:,1],arraySimResponse[:,kk],color='#'+colorsForPlot[kk], label = '$s_'+str(kk+1)+'$') # Simulated Response

    ax.set(xlabel='$y_1$ [-]', 
           ylabel='$m_i$ [g kg$^{-1}$]',
           xlim = [0,1], ylim = [0, 1.1*np.max(arraySimResponse)])     
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    ax.legend()
    if numberOfGases == 3:
        ax2.set(xlabel='$y_2$ [-]',
               ylabel='$m_i$ [g kg$^{-1}$]',
               xlim = [0,1], ylim = [0, 1.1*np.max(arraySimResponse)])     
        ax2.locator_params(axis="x", nbins=4)
        ax2.locator_params(axis="y", nbins=4)
        ax2.legend()
    
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

# Make ternanry/ternary equivalent plots
if numberOfGases == 3 and fixOneGas == False:
    # Loop through all the materials in the array
    sensitiveGroup = {}
    sensitiveResponse = {}
    skewnessResponse = np.zeros(len(sensorID))
    for ii in range(len(sensorID)):
        # Key for dictionary entry
        tempDictKey = 's_{}'.format(ii)
        # Reshape the response for k-means clustering
        reshapedArraySimResponse = np.reshape(arraySimResponse[:,ii],[-1,1])
        # Get the skewness of the sensor resposne
        skewnessResponse = scipy.stats.skew(reshapedArraySimResponse)
        # Perform k-means clustering to separate out the data
        kMeansStruct = KMeans(n_clusters=2,random_state=None).fit(reshapedArraySimResponse)
        # Obtain the group of the sensor (sensitive/non sensitive)
        predictionGroup = kMeansStruct.predict(reshapedArraySimResponse)
        if kMeansStruct.cluster_centers_[0] <= kMeansStruct.cluster_centers_[1]:
            if skewnessResponse < 0.0:
                sensitiveGroup[tempDictKey] = moleFractionRange[predictionGroup==0,:]
                sensitiveResponse[tempDictKey] = reshapedArraySimResponse[predictionGroup==0,:]
                colorGroup = ["#43aa8b","#f94144"]
            else:
                sensitiveGroup[tempDictKey] = moleFractionRange[predictionGroup==1,:]
                sensitiveResponse[tempDictKey] = reshapedArraySimResponse[predictionGroup==1,:]
                colorGroup = ["#f94144","#43aa8b"]
                
        else:
            if skewnessResponse < 0.0:
                sensitiveGroup[tempDictKey] = moleFractionRange[predictionGroup==1,:]
                sensitiveResponse[tempDictKey] = reshapedArraySimResponse[predictionGroup==1,:]
                colorGroup = ["#f94144","#43aa8b"]
            else:
                sensitiveGroup[tempDictKey] = moleFractionRange[predictionGroup==0,:]
                sensitiveResponse[tempDictKey] = reshapedArraySimResponse[predictionGroup==0,:]
                colorGroup = ["#43aa8b","#f94144"]
                
        # Plot raw response in a ternary plot
        fig, tax = ternary.figure(scale=1)
        fig.set_size_inches(4,3.3)
        tax.boundary(linewidth=1.0)
        tax.gridlines(multiple=.2, color="gray")
        tax.scatter(moleFractionRange, marker='o', s=2, c=arraySimResponse[:,ii],
                    vmin=min(arraySimResponse[:,ii]),vmax=max(arraySimResponse[:,ii]),
                    colorbar=True,colormap=plt.cm.PuOr, cmap=plt.cm.PuOr,
                    cbarlabel = '$m_i$ [g kg$^{-1}$]')
        tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
        tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
        tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
        tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
                  offset=0.035,clockwise=True,fontsize=10)
        tax.clear_matplotlib_ticks()
        tax._redraw_labels()
        plt.axis('off')
        if saveFlag:
            # FileName: SensorResponse_<sensorID>_<currentDateTime>_<GitCommitID_Current>
            sensorText = str(sensorID[ii]).replace('[','').replace(']','').replace(' ','-')
            saveFileName = "SensorResponse_" + sensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
            savePath = os.path.join('simulationFigures',saveFileName)
            # Check if inputResources directory exists or not. If not, create the folder
            if not os.path.exists(os.path.join('..','simulationFigures')):
                os.mkdir(os.path.join('..','simulationFigures'))
            plt.savefig (savePath)
        tax.show()

    # Plot the region of sensitivity in a ternary plot overlayed for all the
    # sensors     
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(4,3.3)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=.2, color="gray")
    for ii in range(len(sensitiveGroup)):
        tempDictKey = 's_{}'.format(ii)
        tax.scatter(sensitiveGroup[tempDictKey], marker='o', s=2,
                    color = '#'+colorIntersection[ii], alpha = 0.15)
    tax.scatter(inputMoleFracALL, marker='o', s=20,
            color = '#'+colorsForPlot[0])
    inputMoleFracALL[:,[1,2]]=inputMoleFracALL[:,[2,1]]
    tax.scatter(inputMoleFracALL, marker='o', s=20,
            color = '#'+colorsForPlot[1])
    inputMoleFracALL[:,[0,1]]=inputMoleFracALL[:,[1,0]]
    tax.scatter(inputMoleFracALL, marker='o', s=20,
            color = '#'+colorsForPlot[2])
    tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
              offset=0.035,clockwise=True,fontsize=10)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    plt.axis('off')
    if saveFlag:
        # FileName: SensorRegion_<saveFileSensorText>_<currentDateTime>_<GitCommitID_Current>
        sensorText = str(saveFileSensorText).replace('[','').replace(']','').replace(' ','-')
        saveFileName = "SensorRegion_" + sensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
    tax.show()
    
# Plot the objective function used to evaluate the concentration for individual
# sensors and the total (sum)
if numberOfGases == 3 and fixOneGas == True:
    fig = plt.figure
    ax = plt.subplot(1,3,1)
    for kk in range(arraySimResponse.shape[1]): 
        ax.plot(moleFractionRange[:,0],np.power((arrayTrueResponse[:,kk]
                                                 -arraySimResponse[:,kk])/arrayTrueResponse[:,kk],2),
                color='#'+colorsForPlot[kk], label = '$J_'+str(kk+1)+'$') 
    ax.plot(moleFractionRange[:,0],objFunction,color='#'+colorsForPlot[-1], label = '$\Sigma J_i$')  # Error all sensors
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    ax.set(xlabel='$y_1$ [-]', 
           ylabel='$J$ [-]',
           xlim = [0,1.], ylim = [0, 5])
    ax.legend()
    
    ax = plt.subplot(1,3,2)
    for kk in range(arraySimResponse.shape[1]): 
        ax.plot(moleFractionRange[:,1],np.power((arrayTrueResponse[:,kk]
                                                 -arraySimResponse[:,kk])/arrayTrueResponse[:,kk],2),
                color='#'+colorsForPlot[kk], label = '$J_'+str(kk+1)+'$') 
    ax.plot(moleFractionRange[:,1],objFunction,color='#'+colorsForPlot[-1], label = '$\Sigma J_i$')  # Error all sensors
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    ax.set(xlabel='$y_2$ [-]', 
           ylabel='$J$ [-]',
           xlim = [0,1.], ylim = [0, 5])
    
    ax = plt.subplot(1,3,3)
    for kk in range(arraySimResponse.shape[1]): 
        ax.plot(moleFractionRange[:,2],np.power((arrayTrueResponse[:,kk]
                                                 -arraySimResponse[:,kk])/arrayTrueResponse[:,kk],2),
                color='#'+colorsForPlot[kk], label = '$J_'+str(kk+1)+'$') 
    ax.plot(moleFractionRange[:,2],objFunction,color='#'+colorsForPlot[-1], label = '$\Sigma J_i$')  # Error all sensors
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    ax.set(xlabel='$y_3$ [-]', 
           ylabel='$J$ [-]',
           xlim = [0,1.], ylim = [0, None])
    
    plt.show()