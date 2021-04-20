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
# Plots for the experimental outcome (along with model)
#
# Last modified:
# - 2021-04-20, AK: Implement time-resolved experimental flow rate for DV
# - 2021-04-16, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
from simulateDeadVolume import simulateDeadVolume
from numpy import load
import os
import matplotlib.pyplot as plt
import auxiliaryFunctions
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Save flag for plot
saveFlag = False

# Save file extension
saveFileExtension = ".png"

# Flag to plot dead volume results
flagDeadVolume = True

# Flag to plot simulations
simulateModel = True

# Flag to plot dead volume results
plotFt = True

# Directory of raw data
mainDir = '/Users/ash23win/Google Drive/ERASE/experimental/runData/'

colorsForPlot = ["#E5383B","#E5383B","#B55055","#B55055","#6C757D","#6C757D"]
markerForPlot = ["o","v","o","v","o","v"]

if flagDeadVolume:
    # File name of the experiments
    fileName = ['ZLC_DeadVolume_Exp13A_Output_92e47e7.npz',
                'ZLC_DeadVolume_Exp13B_Output_92e47e7.npz',
                'ZLC_DeadVolume_Exp13C_Output_92e47e7.npz',
                'ZLC_DeadVolume_Exp13D_Output_92e47e7.npz',
                'ZLC_DeadVolume_Exp13E_Output_92e47e7.npz',
                'ZLC_DeadVolume_Exp13F_Output_92e47e7.npz']
    # File with parameter estimates
    simulationDir = '/Users/ash23win/Google Drive/ERASE/simulationResults/'
    fileParameter = 'deadVolumeCharacteristics_20210420_1806_92e47e7.npz'
    modelOutputTemp = load(simulationDir+fileParameter, allow_pickle=True)["modelOutput"]
    x = modelOutputTemp[()]["variable"]
    # Print the objective function and volume from model parameters
    print("Objective Function",round(modelOutputTemp[()]["function"],0))
    print("Model Volume",round(sum(x[0:2]),2))
    # Initialize error for objective function
    # Loop over all available files    
    for ii in range(len(fileName)):
        # Initialize outputs
        moleFracSim = []
        # Path of the file name
        fileToLoad = mainDir + fileName[ii]   
        # Load experimental timeelapsed, molefraction and flowRate
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExp = load(fileToLoad)["moleFrac"].flatten()
        flowRate = load(fileToLoad)["flowRate"].flatten()
        
        # Integration and ode evaluation time
        timeInt = timeElapsedExp
        
        # Print experimental volume 
        print("Experiment",str(ii+1),round(np.trapz(moleFracExp,
                                                    np.multiply(flowRate,timeElapsedExp)),2))
        if simulateModel:
            # Compute the dead volume response using the optimizer parameters
            _ , _ , moleFracSim = simulateDeadVolume(deadVolume_1M = x[0],
                                                                  deadVolume_1D = x[1],
                                                                  numTanks_1M = int(x[2]),
                                                                  numTanks_1D = int(x[3]),
                                                                  splitRatioFactor = x[4],
                                                                  timeInt = timeInt,
                                                                  flowRate = flowRate)

             # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(moleFracSim,
                                                      np.multiply(flowRate,
                                                                  timeElapsedExp)),2))
    
        # Plot the expreimental and model output
        if not plotFt:
            # Linear scale
            fig = plt.figure
            ax1 = plt.subplot(1,2,1)        
            ax1.plot(timeElapsedExp,moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRate),2))+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$t$ [s]', 
                    ylabel='$y_1$ [-]',
                    xlim = [0,50], ylim = [0, 1])         
            ax1.legend()
    
            # Log scale
            ax2 = plt.subplot(1,2,2)        
            ax2.semilogy(timeElapsedExp,moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRate),2))+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$t$ [s]', 
                    xlim = [0,400], ylim = [1e-3, 1])         
            ax2.legend()
            
            #  Save the figure
            if saveFlag:
                # FileName: deadVolumeCharacteristics_<currentDateTime>_<GitCommitID_Current>_<modelFile>
                saveFileName = "deadVolumeCharacteristics_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
                savePath = os.path.join('..','simulationFigures',saveFileName)
                # Check if inputResources directory exists or not. If not, create the folder
                if not os.path.exists(os.path.join('..','simulationFigures')):
                    os.mkdir(os.path.join('..','simulationFigures'))
                plt.savefig (savePath)
        else:
            # Linear scale
            fig = plt.figure
            ax1 = plt.subplot(1,2,1)        
            ax1.plot(np.multiply(flowRate,timeElapsedExp),moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRate),2))+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(np.multiply(flowRate,timeElapsedExp),moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$Ft$ [cc]', 
                    ylabel='$y_1$ [-]',
                    xlim = [0,50], ylim = [0, 1])         
            ax1.legend()
    
            # Log scale
            ax2 = plt.subplot(1,2,2)        
            ax2.semilogy(np.multiply(flowRate,timeElapsedExp),moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRate),2))+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(np.multiply(flowRate,timeElapsedExp),moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$Ft$ [cc]', 
                    xlim = [0,100], ylim = [1e-3, 1])         
            ax2.legend()
            
            #  Save the figure
            if saveFlag:
                # FileName: deadVolumeCharacteristicsFt_<currentDateTime>_<GitCommitID_Current>_<modelFile>
                saveFileName = "deadVolumeCharacteristicsFt_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
                savePath = os.path.join('..','simulationFigures',saveFileName)
                # Check if inputResources directory exists or not. If not, create the folder
                if not os.path.exists(os.path.join('..','simulationFigures')):
                    os.mkdir(os.path.join('..','simulationFigures'))
                plt.savefig (savePath)