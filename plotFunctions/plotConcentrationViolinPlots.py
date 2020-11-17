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
flagComparison = True

# Mole fraction ID
moleFracID = 3
meanMolFrac = [0.001,0.01,0.1,0.25,0.5,0.75,0.90]

# Y limits for the plot
Y_LIMITS = [1e-4,1.]
scaleLog = True

# Legend First
legendText = ["Without Noise", "With Noise"]
legendFlag = True

# Sensor ID
sensorText = ["6/2", "17/16", "17/16", "17/6"]

# Initialize x, y, and type for the plotting
concatenatedX = []
concatenatedY = []
concatenatedType = []

# File to be loaded for the left of violin plot
loadFileName = ["sensitivityAnalysis_6-2_20201113_1450_c9b2a41.npz",
                "sensitivityAnalysis_6-2_20201116_1806_c9b2a41.npz"]
saveFileSensorText = [17, 16]

if flagComparison and len(loadFileName) != 2:
    errorString = "When flagComparison is True, only two files can be loaded for comparison."
    raise Exception(errorString)

# Loop through the different files to generate the violin plot
for kk in range(len(loadFileName)):
    # Initialize x, y, and type for the local loop
    xVar = []
    y1Var = []
    y2Var = []
    typeVar = []

    simResultsFile = os.path.join('..','simulationResults',loadFileName[kk]);
    resultOutput = load(simResultsFile)["arrayConcentration"]
    numberOfGases = load(simResultsFile)["numberOfGases"]
    moleFrac = load(simResultsFile)["moleFractionG1"]
    
    # For the cases where there are two gases 
    if numberOfGases == 2:
        for ii in range(resultOutput.shape[0]):
            # Loop through all the molefractions for comparison
            if flagComparison:
                y1Var = np.concatenate((y1Var,resultOutput[ii,:,2])) # y1
                y2Var = np.concatenate((y2Var,resultOutput[ii,:,3])) # y2
                xVar = xVar + ([str(moleFrac[ii])] * len(resultOutput[ii,:,2])) # x (true mole fraction)
            # Parse the data only for the desired mole fraction
            else:
                if ii == moleFracID:
                    y1Var = np.concatenate((y1Var,resultOutput[ii,:,2])) # y1
                    y2Var = np.concatenate((y2Var,resultOutput[ii,:,3])) # y2            
                    xVar = xVar + ([sensorText[kk]] * len(resultOutput[ii,:,2]))
    # Generate type for comparison
    if flagComparison:
        typeVar = [legendText[kk]] * len(y1Var) # Type - string

    # Concatenate all the data to form a data frame with x, y, and type
    concatenatedX = concatenatedX + xVar
    concatenatedY = np.concatenate((concatenatedY,y1Var))
    concatenatedType = concatenatedType + typeVar
    
    # Reinitialize all the loaded values to empty variable
    simResultsFile = []
    resultOutput = []
    numberOfGases = []
    moleFrac = []

# Generate data frame
# Inclue data type for comparison
if flagComparison:
    df = pd.DataFrame({'x':concatenatedX,
                       'y':concatenatedY,
                       'dataType':concatenatedType})
else:
    df = pd.DataFrame({'x':concatenatedX,
                       'y':concatenatedY})
    
# Plot the figure
sns.set(style="ticks", palette="pastel", color_codes=True)
fig = plt.figure
# Histogram for gas 1
ax1 = plt.subplot(1,1,1)
# Draw a nested violinplot for easier comparison
if flagComparison:
    if scaleLog:
        ax1.set_yscale('log')
    sns.violinplot(data=df, x="x", y="y", hue="dataType", inner = "box",
                   split=True, linewidth=1, palette={"Without Noise": 'r',
                                                     "With Noise": 'g'},
                   scale='width')
    ax1.set(xlabel='$y_1$ [-]', ylabel='${\hat{y}_1}$ [-]', ylim = Y_LIMITS)
    plt.legend(loc='upper left')
    for kk in range(len(meanMolFrac)):
        ax1.axhline(meanMolFrac[kk], linestyle=':', linewidth=1, color = '#c0c0c0')
    if not legendFlag:
        plt.legend([],[], frameon=False)
    ax1.locator_params(axis="y", nbins=4)
# Draw violin plot for compaison of different sensors
else:
    sns.violinplot(data=df, x="x", y="y", inner = "box", linewidth=1,
                   scale='width', palette = ['g', 'g', 'b', 'y'])
    ax1.set(xlabel='Sensor ID [-]', ylabel='${\hat{y}_1}$ [-]', ylim = Y_LIMITS)
    for kk in range(len(meanMolFrac)):
        ax1.axhline(meanMolFrac[kk], linestyle=':', linewidth=1, color = '#646464')
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