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
# - 2021-05-04, AK: Implement plots for ZLC and change DV error computaiton
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
colorsForPlot = ["#E5383B","#CD4448","#B55055",
                 "#9C5D63","#846970","#6C757D"]
# colorsForPlot = ["#E5383B","#6C757D"]
markerForPlot = ["o","v","o","v","o","v"]

if flagDeadVolume:
    # File name of the experiments
    fileName = ['ZLC_DeadVolume_Exp15A_Output_10a7d64.npz',
                'ZLC_DeadVolume_Exp15B_Output_10a7d64.npz',
                'ZLC_DeadVolume_Exp15C_Output_10a7d64.npz']
    # File with parameter estimates
    simulationDir = '/Users/ash23win/Google Drive/ERASE/simulationResults/'
    fileParameter = 'deadVolumeCharacteristics_20210503_0956_cb5686f.npz'
    modelOutputTemp = load(simulationDir+fileParameter, allow_pickle=True)["modelOutput"]
    x = modelOutputTemp[()]["variable"]
    
    numPointsExp = np.zeros(len(fileName))
    for ii in range(len(fileName)): 
        fileToLoad = mainDir + fileName[ii]
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
    
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)

    # Print the objective function and volume from model parameters
    print("Objective Function",round(modelOutputTemp[()]["function"],0))
    print("Model Volume",round(sum(x[0:2]),2))
    computedErrorHigh = 0
    computedErrorLow = 0
    numPoints = 0
    # Create the instance for the plots
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)        
    ax2 = plt.subplot(1,3,2)
    # Initialize error for objective function
    # Loop over all available files    
    for ii in range(len(fileName)):
        # Initialize outputs
        moleFracSim = []
        # Path of the file name
        fileToLoad = mainDir + fileName[ii]   
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
        flowRateTemp = load(fileToLoad)["flowRate"].flatten()
        timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
        moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
        flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
        
        # Integration and ode evaluation time
        timeInt = timeElapsedExp
        
        # Print experimental volume 
        print("Experiment",str(ii+1),round(np.trapz(moleFracExp,
                                                    np.multiply(flowRateExp,timeElapsedExp)),2))
        if simulateModel:
            # Compute the dead volume response using the optimizer parameters
            _ , _ , moleFracSim = simulateDeadVolume(deadVolume_1 = x[0],
                                                    deadVolume_2M = x[1],
                                                    deadVolume_2D = x[2],
                                                    numTanks_1 = int(x[3]),
                                                    flowRate_D = x[4],
                                                    timeInt = timeInt,
                                                    flowRate = flowRateExp,
                                                    expFlag = True)

            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(moleFracSim,
                                                      np.multiply(flowRateExp,
                                                                  timeElapsedExp)),2))

            # Objective function error
            # Find error for mole fraction below a given threshold
            thresholdFactor = 1e-2
            lastIndThreshold = int(np.argwhere(np.array(moleFracExp)>thresholdFactor)[-1])
            # Do downsampling if the number of points in higher and lower
            # compositions does not match
            numPointsConc = np.zeros([2])
            numPointsConc[0] = len(moleFracExp[0:lastIndThreshold])
            numPointsConc[1] = len(moleFracExp[lastIndThreshold:-1])            
            downsampleConc = numPointsConc/np.min(numPointsConc)
            
            # Compute error for higher concentrations
            moleFracHighExp = moleFracExp[0:lastIndThreshold]
            moleFracHighSim = moleFracSim[0:lastIndThreshold]
            computedErrorHigh += np.log(np.sum(np.power(moleFracHighExp[::int(np.round(downsampleConc[0]))] 
                                                        - moleFracHighSim[::int(np.round(downsampleConc[0]))],2)))
            
            # Find scaling factor for lower concentrations
            scalingFactor = int(1/thresholdFactor) # Assumes max composition is one
            # Compute error for lower concentrations
            moleFracLowExp = moleFracExp[lastIndThreshold:-1]*scalingFactor
            moleFracLowSim = moleFracSim[lastIndThreshold:-1]*scalingFactor

            # Compute error for low concentrations (also scaling the compositions)
            computedErrorLow += np.log(np.sum(np.power(moleFracLowExp[::int(np.round(downsampleConc[1]))] 
                                                        - moleFracLowSim[::int(np.round(downsampleConc[1]))],2)))
            
            # Compute the number of points per experiment (accouting for down-
            # sampling in both experiments and high and low compositions
            numPoints += len(moleFracHighExp) + len(moleFracLowExp)

        # Plot the expreimental and model output
        if not plotFt:
            # Linear scale
            ax1.plot(timeElapsedExp,moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$t$ [s]', 
                    ylabel='$y_1$ [-]',
                    xlim = [0,200], ylim = [0, 1])         
            ax1.legend()
    
            # Log scale
            ax2.semilogy(timeElapsedExp,moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$t$ [s]', 
                    xlim = [0,400], ylim = [1e-3, 1])         
            
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
            ax1.plot(np.multiply(flowRateExp,timeElapsedExp),moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(np.multiply(flowRateExp,timeElapsedExp),moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$Ft$ [cc]', 
                    ylabel='$y_1$ [-]',
                    xlim = [0,50], ylim = [0, 1])         
            ax1.legend()
    
            # Log scale
            ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracSim,
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
                
    print(numPoints*(computedErrorHigh+computedErrorLow)/2) 
    
else:
    from simulateCombinedModel import simulateCombinedModel
    
    # Directory of raw data
    mainDir = '/Users/ash23win/Google Drive/ERASE/experimental/runData/'
    # File name of the experiments
    fileName = ['ZLC_ActivatedCarbon_Exp17A_Output_8e60357.npz',
                'ZLC_ActivatedCarbon_Exp17B_Output_8e60357.npz']
    
    # File with parameter estimates
    simulationDir = '/Users/ash23win/Google Drive/ERASE/simulationResults/'
    fileParameter = 'zlcParameters_20210428_1011_10a7d64.npz' # 17A-F
    fileParameter = 'zlcParameters_20210503_1156_cb5686f.npz' # 17A-B
    modelOutputTemp = load(simulationDir+fileParameter, allow_pickle=True)["modelOutput"]
    print("Objective Function",round(modelOutputTemp[()]["function"],0))
    modelNonDim = modelOutputTemp[()]["variable"] 
    # Multiply the paremeters by the reference values
    x = np.multiply(modelNonDim,[10, 1e-5, 50e3, 10, 1e-5, 50e3, 100])
    
    # Ronny AC Data
    # x = [0.44, 3.17e-6, 28.63e3, 6.10, 3.21e-6, 20.37e3,100]

    computedError = 0
    numPoints = 0
    # Create the instance for the plots
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)        
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
                
    # Loop over all available files    
    for ii in range(len(fileName)):
        fileToLoad = mainDir + fileName[ii]   
        
        # Initialize outputs
        moleFracSim = []  
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExp = load(fileToLoad)["moleFrac"].flatten()
        flowRateExp = load(fileToLoad)["flowRate"].flatten()
                
        # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
        timeInt = timeElapsedExp
        
        # Print experimental volume 
        print("Experiment",str(ii+1),round(np.trapz(np.multiply(flowRateExp,moleFracExp),timeElapsedExp),2))

        if simulateModel:
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                        flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                        expFlag = True,
                                                        isothermModel=x[0:-1],
                                                        rateConstant=x[-1])
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  resultMat[0,:]),
                                                        timeElapsedExp),2))
        
            # Objective function error
            computedError += np.log(np.sum(np.power(moleFracExp - moleFracSim,2)))
            numPoints += len(moleFracExp)

        # y - Linear scale
        ax1.semilogy(timeElapsedExp,moleFracExp,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.05) # Experimental response
        if simulateModel:
            ax1.plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],label=str(round(np.mean(flowRateExp),2))+" ccs") # Simulation response    
        ax1.set(xlabel='$t$ [s]', 
                ylabel='$y_1$ [-]',
                xlim = [0,300], ylim = [1e-3, 1])    
        ax1.locator_params(axis="x", nbins=4)
        ax1.legend()
    
        # Ft - Log scale        
        ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracExp,
                      marker = markerForPlot[ii],linewidth = 0,
                      color=colorsForPlot[ii],alpha=0.05) # Experimental response
        if simulateModel:
            ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracSim,
                          color=colorsForPlot[ii],label=str(round(np.mean(flowRateExp),2))+" ccs") # Simulation response
        ax2.set(xlabel='$Ft$ [cc]', 
                xlim = [0,100], ylim = [1e-3, 1])         
        ax2.locator_params(axis="x", nbins=4)
        
        # Flow rates
        ax3.plot(timeElapsedExp,flowRateExp,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.05,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
        if simulateModel:
            ax3.plot(timeElapsedExp,resultMat[3,:]*1e6,
                     color=colorsForPlot[ii]) # Simulation response    
        ax3.set(xlabel='$t$ [s]', 
                ylabel='$F$ [ccs]',
                xlim = [0,300], ylim = [0, 0.5])
        ax3.locator_params(axis="x", nbins=4)
        ax3.locator_params(axis="y", nbins=4)

        #  Save the figure
        if saveFlag:
            # FileName: zlcCharacteristics_<currentDateTime>_<GitCommitID_Current>_<modelFile>
            saveFileName = "zlcCharacteristics_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
            savePath = os.path.join('..','simulationFigures',saveFileName)
            # Check if inputResources directory exists or not. If not, create the folder
            if not os.path.exists(os.path.join('..','simulationFigures')):
                os.mkdir(os.path.join('..','simulationFigures'))
            plt.savefig (savePath)         
        
    plt.show()
    print(numPoints*computedError/2)