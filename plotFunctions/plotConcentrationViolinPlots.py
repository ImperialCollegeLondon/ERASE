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
# Plots to visualize different sensor responses
#
# Last modified:
# - 2020-11-27, AK: Major modification for 3 gas system
# - 2020-11-23, AK: Add standard deviation/CV plotting
# - 2020-11-20, AK: Introduce 3 gas capability
# - 2020-11-18, AK: Changes to data reconciliation and new plots
# - 2020-11-13, AK: Initial creation
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
import os
import pandas as pd
import seaborn as sns 
import auxiliaryFunctions
import matplotlib.pyplot as plt
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Save file extension (png or pdf)
saveFileExtension = ".png"

# Save flag
saveFlag = False

# Flag for comparison
flagComparison = False

# Color combo
colorTemp = ["eac435","345995","03cea4","fb4d3d","ca1551"]
colorForPlot = ["#" + counter for counter in colorTemp] 

# Mole fraction ID
moleFracID = 3
meanMolFrac = [0.00,0.2,0.35,0.5,0.75,0.9]

# Y limits for the plot
Y_LIMITS = [None,None]
scaleLog = True

# Legend First
legendText = ["Without Noise", "With Noise"]
legendFlag = False

# Sensor ID
sensorText = ["12/13/14", "3/4/1", "2/6/8", "0/1/2"]

# Initialize x, y, and type for the plotting
concatenatedX = []
concatenatedX2 = []
concatenatedX3 = []
concatenatedY1 = []
concatenatedY2 = []
concatenatedY3 = []
concatenatedType = []

# File to be loaded for the left of violin plot
loadFileName = ["sensitivityAnalysis_0-6-5_20201127_1217_50a3ed7.npz",
                "sensitivityAnalysis_3-4-1_20201127_1220_50a3ed7.npz"]
saveFileSensorText = [17,15,16,6]

if flagComparison and len(loadFileName) != 2:
    errorString = "When flagComparison is True, only two files can be loaded for comparison."
    raise Exception(errorString)

# Loop through the different files to generate the violin plot
for kk in range(len(loadFileName)):
    # Initialize x, y, and type for the local loop
    xVar = []
    x2Var = []
    x3Var = []
    y1Var = []
    y2Var = []
    y3Var = []
    typeVar = []

    simResultsFile = os.path.join('..','simulationResults',loadFileName[kk]);
    resultOutput = load(simResultsFile)["arrayConcentration"]
    numberOfGases = load(simResultsFile)["numberOfGases"]
    if numberOfGases == 2:
        moleFrac = load(simResultsFile)["trueMoleFrac"]
    elif numberOfGases == 3:
        moleFracTemp = load(simResultsFile)["trueMoleFrac"]
        moleFrac = moleFracTemp[:,0]
        moleFrac2 = moleFracTemp[:,1]
        moleFrac3 = moleFracTemp[:,2]
        
    # Loop through all the molefractions
    for ii in range(resultOutput.shape[0]):
        # For the cases where there are two gases 
        if numberOfGases == 2:
            if resultOutput.shape[2] == 4:
                counterInd = 0
            elif resultOutput.shape[2] == 5:
                counterInd = 1
        # For the cases where there are two gases 
        elif numberOfGases == 3:
            if resultOutput.shape[2] == 6:
                counterInd = 1

        y1Var = np.concatenate((y1Var,resultOutput[ii,:,counterInd+2])) # y1
        y2Var = np.concatenate((y2Var,resultOutput[ii,:,counterInd+3])) # y2
        if numberOfGases == 3:
            y3Var = np.concatenate((y3Var,resultOutput[ii,:,counterInd+4])) # y3
        xVar = xVar + ([str(moleFrac[ii])] * len(resultOutput[ii,:,counterInd+2])) # x (true mole fraction)
        if numberOfGases == 3:
            x2Var = x2Var + ([str(moleFrac2[ii])] * len(resultOutput[ii,:,counterInd+2])) # x (true mole fraction)
            x3Var = x3Var + ([str(moleFrac3[ii])] * len(resultOutput[ii,:,counterInd+2])) # x (true mole fraction)
        if not flagComparison:
            typeVar = typeVar+[sensorText[kk]] * len(resultOutput[ii,:,counterInd+2])
    # Generate type for comparison
    if flagComparison:
        typeVar = [legendText[kk]] * len(y1Var) # Type - string

    # Concatenate all the data to form a data frame with x, y, and type
    concatenatedX = concatenatedX + xVar
    concatenatedY1 = np.concatenate((concatenatedY1,y1Var))
    concatenatedY2 = np.concatenate((concatenatedY2,y2Var))
    if numberOfGases == 3:
        concatenatedY3 = np.concatenate((concatenatedY3,y3Var))
        concatenatedX2 = concatenatedX2 + x2Var
        concatenatedX3 = concatenatedX3 + x3Var

    concatenatedType = concatenatedType + typeVar
    
    # Reinitialize all the loaded values to empty variable
    simResultsFile = []
    resultOutput = []
    moleFrac = []

# Generate panda data frame
# x = molefraction (true)
# y = molefraction (estimated)
# dataType = either sensor id/comparison type
if numberOfGases == 2: 
    df = pd.DataFrame({'x':concatenatedX,
                       'y1':concatenatedY1,
                       'y2':concatenatedY2,
                       'dataType':concatenatedType})
    # Compute the mean and standard deviation
    meanData = df.groupby(['dataType','x'], as_index=False, sort=False).mean() 
    stdData = df.groupby(['dataType','x'], as_index=False, sort=False).std()
    # Coefficient of variation
    cvData = stdData.copy()
    cvData['y1'] = stdData['y1']/meanData['y1']

elif numberOfGases == 3:
    df = pd.DataFrame({'x':concatenatedX,
                       'x2':concatenatedX2,
                       'x3':concatenatedX3,
                       'y1':concatenatedY1,
                       'y2':concatenatedY2,
                       'y3':concatenatedY3,
                       'dataType':concatenatedType})
    # Compute the mean and standard deviation
    meanData = df.groupby(['dataType','x','x2','x3'], as_index=False, sort=False).mean() 
    stdData = df.groupby(['dataType','x','x2','x3'], as_index=False, sort=False).std()
    # Coefficient of variation
    cvData = stdData.copy()
    cvData['y1'] = stdData['y1']/meanData['y1']
    cvData['y2'] = stdData['y2']/meanData['y2']
    cvData['y3'] = stdData['y3']/meanData['y3']

# Plot quantities of interest
# Two gases
if numberOfGases == 2:
    # Plot the figure
    sns.set(style="ticks", palette="pastel", color_codes=True)
    fig = plt.figure
    ax1 = plt.subplot(1,1,1)
    # Draw a nested violinplot for easier comparison
    if flagComparison:
        if scaleLog:
            ax1.set_yscale('log')
        sns.violinplot(data=df, x="x", y="y1", hue="dataType", inner = "box",
                       split=True, linewidth=1, palette={legendText[0]: colorForPlot[0],
                                                         legendText[1]: colorForPlot[1]},
                       scale='width')
        ax1.set(xlabel='$y_1$ [-]', ylabel='${\hat{y}_1}$ [-]', ylim = Y_LIMITS)
        plt.legend(loc='upper left')
        if not legendFlag:
            plt.legend([],[], frameon=False)
    # Draw violin plot for compaison of different sensors
    else:
        sns.violinplot(data=df[df.x == str(meanMolFrac[moleFracID])], 
                       x="dataType", y="y1", inner = "box", linewidth=1,
                       scale='width', palette = colorForPlot[0:len(loadFileName)])
        ax1.set(xlabel='Sensor ID [-]', ylabel='${\hat{y}_1}$ [-]', ylim = Y_LIMITS)
    if flagComparison:
        for kk in range(len(meanMolFrac)):
            ax1.axhline(meanMolFrac[kk], linestyle=':', linewidth=1, color = '#c0c0c0')
    else:
        ax1.axhline(meanMolFrac[moleFracID], linestyle=':', linewidth=1, color = '#c0c0c0')
    ax1.locator_params(axis="y", nbins=4)
    #  Save the figure
    if saveFlag:
        # FileName: SensorViolinPlot_<sensorID>_<noleFrac>_<currentDateTime>_<GitCommitID_Data>>
        saveFileSensorText = str(saveFileSensorText).replace('[','').replace(']','').replace(' ','-').replace(',','')
        saveFileName = "SensorViolinPlot_" + saveFileSensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
    plt.show()

    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure
    # Standard deviation
    ax1 = plt.subplot(1,2,1)
    stdData["x"] = pd.to_numeric(stdData["x"], downcast="float")
    sns.lineplot(data=stdData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = False, markers = ['o']*len(loadFileName),
                 palette = colorForPlot[0:len(loadFileName)])
    ax1.set(xlabel='$y_1$ [-]', 
            ylabel='$\sigma (\hat{y}_1)$ [-]',
            xlim = [0.,1.], ylim = [1e-6,1.])
    ax1.set_yscale('log')
    ax1.locator_params(axis="x", nbins=4)
    if len(loadFileName) > 1:
        plt.legend(loc='best')
    else:
        plt.legend([],[], frameon=False)
    
    # CV
    ax2 = plt.subplot(1,2,2)
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = False, markers = ['o']*len(loadFileName),
                 palette = colorForPlot[0:len(loadFileName)])
    ax2.set(xlabel='$y_1$ [-]', 
            ylabel='$CV (\hat{y}_1)$ [-]',
            xlim = [0.,1.], ylim = [1e-5,1.])
    ax2.locator_params(axis="x", nbins=4)
    ax2.set_yscale('log')
    if not legendFlag:
        plt.legend([],[], frameon=False)
    if saveFlag:
        # FileName: SensorViolinPlot_<sensorID>_<noleFrac>_<currentDateTime>_<GitCommitID_Data>>
        saveFileSensorText = str(saveFileSensorText).replace('[','').replace(']','').replace(' ','-').replace(',','')
        saveFileName = "SensorStdCV_" + saveFileSensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
    plt.show()

# Three gases
if numberOfGases == 3:
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = False, markers = ['o']*len(loadFileName),
                 palette = colorForPlot[0:len(loadFileName)])
    ax1.set(xlabel='$y_1$ [-]', 
            ylabel='$CV (\hat{y}_1)$ [-]',
            xlim = [0.,1.], ylim = [1e-5,1.])
    ax1.locator_params(axis="x", nbins=4)
    ax1.set_yscale('log')
    if not legendFlag:
        plt.legend([],[], frameon=False)
    ax2 = plt.subplot(1,3,2)
    cvData["x2"] = pd.to_numeric(cvData["x2"], downcast="float")
    sns.lineplot(data=cvData, x='x2', y='y2', hue='dataType', style='dataType',
                 dashes = False, markers = ['o']*len(loadFileName),
                 palette = colorForPlot[0:len(loadFileName)])
    ax2.set(xlabel='$y_2$ [-]', 
            ylabel='$CV (\hat{y}_2)$ [-]',
            xlim = [0.,1.], ylim = [1e-5,1.])
    ax2.locator_params(axis="x", nbins=4)
    ax2.set_yscale('log')
    if not legendFlag:
        plt.legend([],[], frameon=False)
    ax3 = plt.subplot(1,3,3)
    cvData["x3"] = pd.to_numeric(cvData["x3"], downcast="float")
    sns.lineplot(data=cvData, x='x3', y='y3', hue='dataType', style='dataType',
                 dashes = False, markers = ['o']*len(loadFileName),
                 palette = colorForPlot[0:len(loadFileName)])
    ax3.set(xlabel='$y_1$ [-]', 
            ylabel='$CV (\hat{y}_3)$ [-]',
            xlim = [0.,1.], ylim = [1e-5,1.])
    ax3.locator_params(axis="x", nbins=4)
    ax3.set_yscale('log')
    if not legendFlag:
        plt.legend([],[], frameon=False)

    if saveFlag:
        # FileName: SensorViolinPlot_<sensorID>_<noleFrac>_<currentDateTime>_<GitCommitID_Data>>
        saveFileSensorText = str(saveFileSensorText).replace('[','').replace(']','').replace(' ','-').replace(',','')
        saveFileName = "SensorStdCV_" + saveFileSensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
    plt.show()