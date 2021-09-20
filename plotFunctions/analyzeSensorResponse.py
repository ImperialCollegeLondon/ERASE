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
from generateTrueSensorResponse import generateTrueSensorResponse
from simulateSensorArray import simulateSensorArray
import os
import pandas as pd
import ternary
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
import auxiliaryFunctions

os.chdir("..")

# Save flag for figure
saveFlag = False

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Plotting colors
colorsForPlot = ["eac435","345995","03cea4","fb4d3d","ca1551"]

# Number of molefractions
numMolFrac= 101

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15]);

# Number of Adsorbents
numberOfAdsorbents = 20

# Number of Gases
numberOfGases = 2

# Multiplier Error
multiplierError = [1., 1.]

# Sensor ID
sensorID = np.array([6,2])

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Simulate the sensor response for all possible concentrations
moleFractionRange = {}
moleFractionRange_y1 = {}
moleFractionRange_y2 = {}
# Two gases
if numberOfGases == 2:
    moleFractionRange['y3_0'] = np.array([np.linspace(0,1,numMolFrac), 1 - np.linspace(0,1,numMolFrac)]).T
# Three gases - done by fixing one gas and going through combinations of 
# y1 and y2 
elif numberOfGases == 3:
    thirdGasMoleFrac = np.linspace(0,1.,numMolFrac)
    for ii in range(len(thirdGasMoleFrac)):
        tempDictKey = 'y3_{}'.format(ii)
        remainingMoleFrac = 1. - thirdGasMoleFrac[ii]
        moleFractionRange[tempDictKey] = np.array([np.linspace(0,remainingMoleFrac,numMolFrac), 
                                                   remainingMoleFrac - np.linspace(0,remainingMoleFrac,numMolFrac),
                                                   np.tile(thirdGasMoleFrac[ii],numMolFrac)]).T
    
    secondGasMoleFrac = np.linspace(0,1.,numMolFrac)
    for ii in range(len(secondGasMoleFrac)):
        tempDictKey = 'y2_{}'.format(ii)
        remainingMoleFrac = 1. - secondGasMoleFrac[ii]
        moleFractionRange_y2[tempDictKey] = np.array([np.linspace(0,remainingMoleFrac,numMolFrac), 
                                                      np.tile(secondGasMoleFrac[ii],numMolFrac),
                                                      remainingMoleFrac - np.linspace(0,remainingMoleFrac,numMolFrac)]).T

    firstGasMoleFrac = np.linspace(0,1.,numMolFrac)
    for ii in range(len(firstGasMoleFrac)):
        tempDictKey = 'y1_{}'.format(ii)
        remainingMoleFrac = 1. - firstGasMoleFrac[ii]
        moleFractionRange_y1[tempDictKey] = np.array([np.tile(firstGasMoleFrac[ii],numMolFrac),
                                                      np.linspace(0,remainingMoleFrac,numMolFrac), 
                                                      remainingMoleFrac - np.linspace(0,remainingMoleFrac,numMolFrac)]).T
        
# Get the simulated sensor response, condition number and the derivative of 
# the mole fraction wrt the sensor response
arraySimResponse = {}
firstDerivativeSimResponse_y1 = {}
firstDerivativeSimResponse_y2 = {}
conditionNumber_y1 = {}
conditionNumber_y2 = {}
# Loop through all variables
for ii in range(len(moleFractionRange)):
    tempDictKey = 'y3_{}'.format(ii)   
    # Temporary variable
    moleFractionRangeTemp = moleFractionRange[tempDictKey]
    arraySimResponseTemp = np.zeros([moleFractionRangeTemp.shape[0],
                                              sensorID.shape[0]])
    firstDerivativeSimResponse_y1Temp = np.zeros([moleFractionRangeTemp.shape[0],
                                              sensorID.shape[0]])
    firstDerivativeSimResponse_y2Temp = np.zeros([moleFractionRangeTemp.shape[0],
                                              sensorID.shape[0]])
    conditionNumber_y1Temp = np.zeros([moleFractionRangeTemp.shape[0],
                                              sensorID.shape[0]])
    conditionNumber_y2Temp = np.zeros([moleFractionRangeTemp.shape[0],
                                              sensorID.shape[0]])
    # Loop through all mole fractions (y1 and y2) to get sensor response
    for kk in range(moleFractionRangeTemp.shape[0]):
        arraySimResponseTemp[kk,:] = simulateSensorArray(sensorID, pressureTotal, 
                                                    temperature, np.array([moleFractionRangeTemp[kk,:]])) * multiplierError

    # Loop through all sensor responses to get the derivative and the condition
    # number
    for jj in range(arraySimResponseTemp.shape[1]):
        firstDerivativeSimResponse_y1Temp[:,jj] = np.gradient(moleFractionRangeTemp[:,0],
                                                              arraySimResponseTemp[:,jj])
        # Condition number
        conditionNumber_y1Temp[:,jj] = np.abs(np.multiply(np.divide(firstDerivativeSimResponse_y1Temp[:,jj],
                                                                    moleFractionRangeTemp[:,0]),
                                                          arraySimResponseTemp[:,jj]))
        firstDerivativeSimResponse_y2Temp[:,jj] = np.gradient(moleFractionRangeTemp[:,1],
                                                              arraySimResponseTemp[:,jj])    
        # Condition number
        conditionNumber_y2Temp[:,jj] = np.abs(np.multiply(np.divide(firstDerivativeSimResponse_y2Temp[:,jj],
                                                                    moleFractionRangeTemp[:,1]),
                                                          arraySimResponseTemp[:,jj]))
    # Save the sensor responseand the derivative to a dictionary    
    arraySimResponse[tempDictKey] = arraySimResponseTemp
    firstDerivativeSimResponse_y1[tempDictKey] = firstDerivativeSimResponse_y1Temp
    firstDerivativeSimResponse_y2[tempDictKey] = firstDerivativeSimResponse_y2Temp
    conditionNumber_y1[tempDictKey] = conditionNumber_y1Temp
    conditionNumber_y2[tempDictKey] = conditionNumber_y2Temp
    arraySimResponseTemp = []
    firstDerivativeSimResponse_y1Temp = []
    firstDerivativeSimResponse_y2Temp = []
    conditionNumber_y1Temp = []
    conditionNumber_y2Temp = []
   
# For three gases
if numberOfGases == 3:       
    arraySimResponse_y2 = {}
    firstDerivativeSimResponse_y31 = {}
    firstDerivativeSimResponse_y3 = {}
    for ii in range(len(moleFractionRange_y2)):        
        tempDictKey = 'y2_{}'.format(ii)   
        # Temporary variable
        moleFractionRangeTemp = moleFractionRange_y2[tempDictKey]
        arraySimResponseTemp = np.zeros([moleFractionRangeTemp.shape[0],
                                                  sensorID.shape[0]])        
        firstDerivativeSimResponse_y31Temp = np.zeros([moleFractionRangeTemp.shape[0],
                                              sensorID.shape[0]])    
        firstDerivativeSimResponse_y3Temp = np.zeros([moleFractionRangeTemp.shape[0],
                                              sensorID.shape[0]])    
        
        # Loop through all mole fractions (y1 and y2) to get sensor response
        for kk in range(moleFractionRangeTemp.shape[0]):
            arraySimResponseTemp[kk,:] = simulateSensorArray(sensorID, pressureTotal, 
                                                        temperature, np.array([moleFractionRangeTemp[kk,:]])) * multiplierError
    
        # Loop through all sensor responses to get the derivative
        for jj in range(arraySimResponseTemp.shape[1]):
            firstDerivativeSimResponse_y31Temp[:,jj] = np.gradient(moleFractionRangeTemp[:,0],arraySimResponseTemp[:,jj])

            firstDerivativeSimResponse_y3Temp[:,jj] = np.gradient(moleFractionRangeTemp[:,2],
                                                                  arraySimResponseTemp[:,jj])
    
        # Save the sensor responseand the derivative to a dictionary    
        arraySimResponse_y2[tempDictKey] = arraySimResponseTemp
        firstDerivativeSimResponse_y31[tempDictKey] = firstDerivativeSimResponse_y31Temp
        firstDerivativeSimResponse_y3[tempDictKey] = firstDerivativeSimResponse_y3Temp
        arraySimResponseTemp = []
        firstDerivativeSimResponse_y31Temp = []
        firstDerivativeSimResponse_y3Temp = []
    
    
    # arraySimResponse_y1 = {}
    # firstDerivativeSimResponse_y12 = {}
    # firstDerivativeSimResponse_y13 = {}
    # for ii in range(len(moleFractionRange_y1)):        
    #     tempDictKey = 'y1_{}'.format(ii)   
    #     # Temporary variable
    #     moleFractionRangeTemp = moleFractionRange_y1[tempDictKey]
    #     arraySimResponseTemp = np.zeros([moleFractionRangeTemp.shape[0],
    #                                               sensorID.shape[0]])        
    #     firstDerivativeSimResponse_y12Temp = np.zeros([moleFractionRangeTemp.shape[0],
    #                                           sensorID.shape[0]])    
    #     firstDerivativeSimResponse_y13Temp = np.zeros([moleFractionRangeTemp.shape[0],
    #                                           sensorID.shape[0]])    
        
    #     # Loop through all mole fractions (y1 and y2) to get sensor response
    #     for kk in range(moleFractionRangeTemp.shape[0]):
    #         arraySimResponseTemp[kk,:] = simulateSensorArray(sensorID, pressureTotal, 
    #                                                     temperature, np.array([moleFractionRangeTemp[kk,:]])) * multiplierError
    
    #     # Loop through all sensor responses to get the derivative
    #     for jj in range(arraySimResponseTemp.shape[1]):
    #         firstDerivativeSimResponse_y12Temp[:,jj] = np.gradient(moleFractionRangeTemp[:,1],
    #                                                               arraySimResponseTemp[:,jj])

    #         firstDerivativeSimResponse_y13Temp[:,jj] = np.gradient(moleFractionRangeTemp[:,2],
    #                                                               arraySimResponseTemp[:,jj])
    
    #     # Save the sensor responseand the derivative to a dictionary    
    #     arraySimResponse_y1[tempDictKey] = arraySimResponseTemp
    #     firstDerivativeSimResponse_y12[tempDictKey] = firstDerivativeSimResponse_y12Temp
    #     firstDerivativeSimResponse_y13[tempDictKey] = firstDerivativeSimResponse_y13Temp
    #     arraySimResponseTemp = []
    #     firstDerivativeSimResponse_y12Temp = []
    #     firstDerivativeSimResponse_y13Temp = []
        
# Plot the sensor response and the derivative
# Two gases
if numberOfGases == 2:
    fig = plt.figure
    # Parse out mole fraction and sensor response
    arraySimResponseTemp = arraySimResponse['y3_0']
    moleFractionRangeTemp = moleFractionRange['y3_0']
    firstDerivativeSimResponse_y1Temp = firstDerivativeSimResponse_y1['y3_0']
    conditionNumber_y1Temp = conditionNumber_y1['y3_0']
    
    ax1 = plt.subplot(1,3,1)
    # Loop through all sensors and plot the sensor response
    for kk in range(arraySimResponseTemp.shape[1]):
        ax1.plot(arraySimResponseTemp[:,kk],moleFractionRangeTemp[:,0],
                 color='#'+colorsForPlot[kk], label = '$s_'+str(kk+1)+'$') # Simulated Response
    ax1.set(ylabel='$y_1$ [-]', 
       xlabel='$m_i$ [g kg$^{-1}$]',
       ylim = [0,1], xlim = [0, 1.1*np.max(arraySimResponseTemp)])     
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.legend()
        
    ax2 = plt.subplot(1,3,2)
    # Loop through all sensors and plot the sensor derivative
    for kk in range(arraySimResponseTemp.shape[1]):
        ax2.plot(arraySimResponseTemp[:,kk],conditionNumber_y1Temp[:,kk],
                 color='#'+colorsForPlot[kk], label = '$s_'+str(kk+1)+'$') # Simulated Response
    ax2.set(xlabel='$m_i$ [g kg$^{-1}$]', 
       ylabel='$\chi$ [-]',
       xlim = [0,1.1*np.max(arraySimResponseTemp)])     
    ax2.locator_params(axis="x", nbins=4)
    ax2.locator_params(axis="y", nbins=4)
    ax2.legend()

    ax3 = plt.subplot(1,3,3)
    # Loop through all sensors and plot the sensor derivative
    for kk in range(arraySimResponseTemp.shape[1]):
        ax3.plot(moleFractionRangeTemp[:,0],conditionNumber_y1Temp[:,kk],
                 color='#'+colorsForPlot[kk], label = '$s_'+str(kk+1)+'$') # Simulated Response
    ax3.set(xlabel='$y_1$ [-]', 
       ylabel='$\chi$ [-]',
       xlim = [0,1])     
    ax3.locator_params(axis="x", nbins=4)
    ax3.locator_params(axis="y", nbins=4)
    ax3.legend()
    
    plt.show()

# Three gases    
elif numberOfGases == 3:
    # Sensor response - y3
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(4,3.3)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=.2, color="gray")
    currentMin = np.Inf
    currentMax = -np.Inf
    for ii in range(len(moleFractionRange)-1):
        tempDictKey = 'y3_{}'.format(ii)
        arraySimResponseTemp = arraySimResponse[tempDictKey]
        responseMin = min(arraySimResponseTemp)
        responseMax = max(arraySimResponseTemp)
        currentMin = min([currentMin,responseMin])
        currentMax = max([currentMax,responseMax])
        
    for ii in range(len(moleFractionRange)-1):
        tempDictKey = 'y3_{}'.format(ii)
        moleFractionRangeTemp = moleFractionRange[tempDictKey]
        arraySimResponseTemp = arraySimResponse[tempDictKey]
        if ii == len(moleFractionRange)-2:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=arraySimResponseTemp,
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
                        colorbar = True, colormap=plt.cm.PuOr)
        else:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=arraySimResponseTemp,
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

        moleFractionRangeTemp = []
        arraySimResponseTemp = []
    
    tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
              offset=0.035,clockwise=True,fontsize=10)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    plt.axis('off')
    tax.show()
    
    # Sensor response - y2
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(4,3.3)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=.2, color="gray")
    currentMin = np.Inf
    currentMax = -np.Inf
    for ii in range(len(moleFractionRange_y2)-1):
        tempDictKey = 'y2_{}'.format(ii)
        arraySimResponseTemp = arraySimResponse_y2[tempDictKey]
        responseMin = min(arraySimResponseTemp)
        responseMax = max(arraySimResponseTemp)
        currentMin = min([currentMin,responseMin])
        currentMax = max([currentMax,responseMax])
        
    for ii in range(len(moleFractionRange)-1):
        tempDictKey = 'y2_{}'.format(ii)
        moleFractionRangeTemp = moleFractionRange_y2[tempDictKey]
        arraySimResponseTemp = arraySimResponse_y2[tempDictKey]
        if ii == len(moleFractionRange)-2:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=arraySimResponseTemp,
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
                        colorbar = True, colormap=plt.cm.PuOr)
        else:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=arraySimResponseTemp,
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

        moleFractionRangeTemp = []
        arraySimResponseTemp = []
    
    tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
              offset=0.035,clockwise=True,fontsize=10)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    plt.axis('off')
    tax.show()
    
    # # Sensor response - y2
    # fig, tax = ternary.figure(scale=1)
    # fig.set_size_inches(4,3.3)
    # tax.boundary(linewidth=1.0)
    # tax.gridlines(multiple=.2, color="gray")
    # currentMin = np.Inf
    # currentMax = -np.Inf
    # for ii in range(len(moleFractionRange_y1)-1):
    #     tempDictKey = 'y1_{}'.format(ii)
    #     arraySimResponseTemp = arraySimResponse_y1[tempDictKey]
    #     responseMin = min(arraySimResponseTemp)
    #     responseMax = max(arraySimResponseTemp)
    #     currentMin = min([currentMin,responseMin])
    #     currentMax = max([currentMax,responseMax])
        
    # for ii in range(len(moleFractionRange)-1):
    #     tempDictKey = 'y1_{}'.format(ii)
    #     moleFractionRangeTemp = moleFractionRange_y1[tempDictKey]
    #     arraySimResponseTemp = arraySimResponse_y1[tempDictKey]
    #     if ii == len(moleFractionRange)-2:
    #         tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=arraySimResponseTemp,
    #                     vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
    #                     colorbar = True, colormap=plt.cm.PuOr)
    #     else:
    #         tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=arraySimResponseTemp,
    #                     vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

    #     moleFractionRangeTemp = []
    #     arraySimResponseTemp = []
    
    # tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    # tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    # tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    # tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
    #           offset=0.035,clockwise=True,fontsize=10)
    # tax.clear_matplotlib_ticks()
    # tax._redraw_labels()
    # plt.axis('off')
    # tax.show()

    # Derivative - dy1/dm
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(4,3.3)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=.2, color="gray")
    currentMin = np.Inf
    currentMax = -np.Inf
    for ii in range(len(moleFractionRange)-1):
        tempDictKey = 'y3_{}'.format(ii)
        firstDerivativeSimResponse_y1Temp = firstDerivativeSimResponse_y1[tempDictKey]
        firstDerivativeMin = min(firstDerivativeSimResponse_y1Temp[:,0])
        firstDerivativeMax = max(firstDerivativeSimResponse_y1Temp[:,0])
        currentMin = min([currentMin,firstDerivativeMin])
        currentMax = max([currentMax,firstDerivativeMax])
        
    for ii in range(len(moleFractionRange)-1):
        tempDictKey = 'y3_{}'.format(ii)
        moleFractionRangeTemp = moleFractionRange[tempDictKey]
        firstDerivativeSimResponse_y1Temp = firstDerivativeSimResponse_y1[tempDictKey]
        if ii == len(moleFractionRange)-2:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y1Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
                        colorbar = True, colormap=plt.cm.PuOr)
        else:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y1Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

        moleFractionRangeTemp = []
        firstDerivativeSimResponse_y1Temp = []
    
    tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
              offset=0.035,clockwise=True,fontsize=10)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    plt.axis('off')
    tax.show()
 
    # Derivative - dy2/dm
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(4,3.3)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=.2, color="gray")
    currentMin = np.Inf
    currentMax = -np.Inf
    for ii in range(len(moleFractionRange)-1):
        tempDictKey = 'y3_{}'.format(ii)
        firstDerivativeSimResponse_y2Temp = firstDerivativeSimResponse_y2[tempDictKey]
        firstDerivativeMin = min(firstDerivativeSimResponse_y2Temp[:,0])
        firstDerivativeMax = max(firstDerivativeSimResponse_y2Temp[:,0])
        currentMin = min([currentMin,firstDerivativeMin])
        currentMax = max([currentMax,firstDerivativeMax])
        
    for ii in range(len(moleFractionRange)-1):
        tempDictKey = 'y3_{}'.format(ii)
        moleFractionRangeTemp = moleFractionRange[tempDictKey]
        firstDerivativeSimResponse_y2Temp = firstDerivativeSimResponse_y2[tempDictKey]
        if ii == len(moleFractionRange)-2:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y2Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
                        colorbar = True, colormap=plt.cm.PuOr)
        else:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y2Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

        moleFractionRangeTemp = []
        firstDerivativeSimResponse_y2Temp = []
    
    tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
              offset=0.035,clockwise=True,fontsize=10)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    plt.axis('off')
    tax.show()
    
    # Derivative - dy3/dm
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(4,3.3)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=.2, color="gray")
    currentMin = np.Inf
    currentMax = -np.Inf
    for ii in range(len(moleFractionRange_y2)-1):
        tempDictKey = 'y2_{}'.format(ii)
        firstDerivativeSimResponse_y3Temp = firstDerivativeSimResponse_y3[tempDictKey]
        firstDerivativeMin = min(firstDerivativeSimResponse_y3Temp[:,0])
        firstDerivativeMax = max(firstDerivativeSimResponse_y3Temp[:,0])
        currentMin = min([currentMin,firstDerivativeMin])
        currentMax = max([currentMax,firstDerivativeMax])
        
    for ii in range(len(moleFractionRange_y2)-1):
        tempDictKey = 'y2_{}'.format(ii)
        moleFractionRangeTemp = moleFractionRange_y2[tempDictKey]
        firstDerivativeSimResponse_y3Temp = firstDerivativeSimResponse_y3[tempDictKey]
        if ii == len(moleFractionRange)-2:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y3Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
                        colorbar = True, colormap=plt.cm.PuOr)
        else:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y3Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

        moleFractionRangeTemp = []
        firstDerivativeSimResponse_y3Temp = []
    
    tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
              offset=0.035,clockwise=True,fontsize=10)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    plt.axis('off')
    tax.show()
    
    # Derivative - dy3/dm
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(4,3.3)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=.2, color="gray")
    currentMin = np.Inf
    currentMax = -np.Inf
    for ii in range(len(moleFractionRange_y2)-1):
        tempDictKey = 'y2_{}'.format(ii)
        firstDerivativeSimResponse_y31Temp = firstDerivativeSimResponse_y31[tempDictKey]
        firstDerivativeMin = min(firstDerivativeSimResponse_y31Temp[:,0])
        firstDerivativeMax = max(firstDerivativeSimResponse_y31Temp[:,0])
        currentMin = min([currentMin,firstDerivativeMin])
        currentMax = max([currentMax,firstDerivativeMax])
        
    for ii in range(len(moleFractionRange_y2)-1):
        tempDictKey = 'y2_{}'.format(ii)
        moleFractionRangeTemp = moleFractionRange_y2[tempDictKey]
        firstDerivativeSimResponse_y31Temp = firstDerivativeSimResponse_y31[tempDictKey]
        if ii == len(moleFractionRange)-2:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y31Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
                        colorbar = True, colormap=plt.cm.PuOr)
        else:
            tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y31Temp[:,0],
                        vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

        moleFractionRangeTemp = []
        firstDerivativeSimResponse_y31Temp = []
    
    tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
              offset=0.035,clockwise=True,fontsize=10)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    plt.axis('off')
    tax.show()
    
    # # Derivative - dy3/dm
    # fig, tax = ternary.figure(scale=1)
    # fig.set_size_inches(4,3.3)
    # tax.boundary(linewidth=1.0)
    # tax.gridlines(multiple=.2, color="gray")
    # currentMin = np.Inf
    # currentMax = -np.Inf
    # for ii in range(len(moleFractionRange_y1)-1):
    #     tempDictKey = 'y1_{}'.format(ii)
    #     firstDerivativeSimResponse_y12Temp = firstDerivativeSimResponse_y12[tempDictKey]
    #     firstDerivativeMin = min(firstDerivativeSimResponse_y12Temp[:,0])
    #     firstDerivativeMax = max(firstDerivativeSimResponse_y12Temp[:,0])
    #     currentMin = min([currentMin,firstDerivativeMin])
    #     currentMax = max([currentMax,firstDerivativeMax])
        
    # for ii in range(len(moleFractionRange_y1)-1):
    #     tempDictKey = 'y1_{}'.format(ii)
    #     moleFractionRangeTemp = moleFractionRange_y1[tempDictKey]
    #     firstDerivativeSimResponse_y12Temp = firstDerivativeSimResponse_y12[tempDictKey]
    #     if ii == len(moleFractionRange)-2:
    #         tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y12Temp[:,0],
    #                     vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
    #                     colorbar = True, colormap=plt.cm.PuOr)
    #     else:
    #         tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y12Temp[:,0],
    #                     vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

    #     moleFractionRangeTemp = []
    #     firstDerivativeSimResponse_y12Temp = []
    
    # tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    # tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    # tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    # tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
    #           offset=0.035,clockwise=True,fontsize=10)
    # tax.clear_matplotlib_ticks()
    # tax._redraw_labels()
    # plt.axis('off')
    # tax.show()
    
    # # Derivative - dy3/dm
    # fig, tax = ternary.figure(scale=1)
    # fig.set_size_inches(4,3.3)
    # tax.boundary(linewidth=1.0)
    # tax.gridlines(multiple=.2, color="gray")
    # currentMin = np.Inf
    # currentMax = -np.Inf
    # for ii in range(len(moleFractionRange_y1)-1):
    #     tempDictKey = 'y1_{}'.format(ii)
    #     firstDerivativeSimResponse_y13Temp = firstDerivativeSimResponse_y13[tempDictKey]
    #     firstDerivativeMin = min(firstDerivativeSimResponse_y13Temp[:,0])
    #     firstDerivativeMax = max(firstDerivativeSimResponse_y13Temp[:,0])
    #     currentMin = min([currentMin,firstDerivativeMin])
    #     currentMax = max([currentMax,firstDerivativeMax])
        
    # for ii in range(len(moleFractionRange_y1)-1):
    #     tempDictKey = 'y1_{}'.format(ii)
    #     moleFractionRangeTemp = moleFractionRange_y1[tempDictKey]
    #     firstDerivativeSimResponse_y13Temp = firstDerivativeSimResponse_y13[tempDictKey]
    #     if ii == len(moleFractionRange)-2:
    #         tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y13Temp[:,0],
    #                     vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr,
    #                     colorbar = True, colormap=plt.cm.PuOr)
    #     else:
    #         tax.scatter(moleFractionRangeTemp, marker='o', s=2, c=firstDerivativeSimResponse_y13Temp[:,0],
    #                     vmin=currentMin,vmax=currentMax,cmap=plt.cm.PuOr)

    #     moleFractionRangeTemp = []
    #     firstDerivativeSimResponse_y13Temp = []
    
    # tax.left_axis_label("$y_2$ [-]",offset=0.20,fontsize=10)
    # tax.right_axis_label("$y_1$ [-]",offset=0.20,fontsize=10)
    # tax.bottom_axis_label("$y_3$ [-]",offset=0.20,fontsize=10)
    # tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f",
    #           offset=0.035,clockwise=True,fontsize=10)
    # tax.clear_matplotlib_ticks()
    # tax._redraw_labels()
    # plt.axis('off')
    # tax.show()