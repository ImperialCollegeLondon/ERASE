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
from scipy.interpolate import interp1d
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
simulateModel = False

# Flag to plot dead volume results
plotFt = False

# Directory of raw data
mainDir = '/Users/ash23win/Google Drive/ERASE/experimental/runData/'

colorsForPlot = ["#E5383B","#B55055","#6C757D"]

if flagDeadVolume:
    # File name of the experiments
    fileName = ['ZLC_DeadVolume_Exp12D_Output_10f6fad.npz',
                'ZLC_DeadVolume_Exp12E_Output_10f6fad.npz',
                'ZLC_DeadVolume_Exp12F_Output_10f6fad.npz']
    
    # File with parameter estimates
    simulationDir = '/Users/ash23win/Google Drive/ERASE/simulationResults/'
    fileParameter = 'deadVolumeCharacteristics_20210415_1657_a72563b.npz'
    modelOutputTemp = load(simulationDir+fileParameter, allow_pickle=True)["modelOutput"]
    x = modelOutputTemp[()]["variable"]
    # Print the objective function and volume from model parameters
    print("Objective Function",round(modelOutputTemp[()]["function"],0))
    print("Model Volume",round(sum(x[0:4]),2))
    # Initialize error for objective function
    # Loop over all available files    
    for ii in range(len(fileName)):
        # Initialize outputs
        timeSimOut = []
        moleFracOut = []
        moleFracSim = []
        # Path of the file name
        fileToLoad = mainDir + fileName[ii]   
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExp = load(fileToLoad)["moleFrac"].flatten()
        # Parse out flow rate of the experiment
        # Obtain the mean and round it to the 2 decimal to be used in the 
        # simulation
        flowRate = round(np.mean(load(fileToLoad)["flowRate"]),2)
        
        # Print experimental volume 
        print("Experiment",str(ii+1),round(np.trapz(moleFracExp,
                                                    np.multiply(load(fileToLoad)["flowRate"].flatten(),
                                                                timeElapsedExp)),2))
        if simulateModel:
            # Compute the dead volume response using the optimizer parameters
            timeSimOut , _ , moleFracOut = simulateDeadVolume(deadVolume_1M = x[0],
                                                                  deadVolume_1D = x[1],
                                                                  deadVolume_2M = x[2],
                                                                  deadVolume_2D = x[3],
                                                                  numTanks_1M = int(x[4]),
                                                                  numTanks_1D = int(x[5]),
                                                                  numTanks_2M = int(x[6]),
                                                                  numTanks_2D = int(x[7]),
                                                                  splitRatio_1 = x[8],
                                                                  splitRatio_2 = x[9],
                                                                  flowRate = flowRate)
        
            # Interpolate simulation data (generate function)
            interpSim = interp1d(timeSimOut, moleFracOut)    
            
            # Find the interpolated simulation mole fraction at times corresponding to 
            # the experimental ones
            moleFracSim = interpSim(timeElapsedExp)
            
             # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(moleFracSim,
                                                      np.multiply(load(fileToLoad)["flowRate"].flatten(),
                                                                  timeElapsedExp)),2))
    
        # Plot the expreimental and model output
        if not plotFt:
            # Linear scale
            fig = plt.figure
            ax1 = plt.subplot(1,2,1)        
            ax1.plot(timeElapsedExp,moleFracExp,
                          'o',color=colorsForPlot[ii],alpha=0.2,label=str(flowRate)+" ccs") # Experimental response
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
                          'o',color=colorsForPlot[ii],alpha=0.2,label=str(flowRate)+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$t$ [s]', 
                    xlim = [0,250], ylim = [1e-3, 1])         
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
            ax1.plot(np.multiply(load(fileToLoad)["flowRate"].flatten(),timeElapsedExp),moleFracExp,
                          'o',color=colorsForPlot[ii],alpha=0.2,label=str(flowRate)+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(flowRate*timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$Ft$ [cc]', 
                    ylabel='$y_1$ [-]',
                    xlim = [0,50], ylim = [0, 1])         
            ax1.legend()
    
            # Log scale
            ax2 = plt.subplot(1,2,2)        
            ax2.semilogy(np.multiply(load(fileToLoad)["flowRate"].flatten(),timeElapsedExp),moleFracExp,
                          'o',color=colorsForPlot[ii],alpha=0.2,label=str(flowRate)+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(flowRate*timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$Ft$ [cc]', 
                    xlim = [0,30], ylim = [1e-2, 1])         
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