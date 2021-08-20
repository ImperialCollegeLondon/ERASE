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
# - 2021-08-20, AK: Change definition of rate constants
# - 2021-07-03, AK: Remove threshold factor
# - 2021-07-01, AK: Cosmetic changes
# - 2021-05-14, AK: Fixes and structure changes
# - 2021-05-14, AK: Improve plotting capabilities
# - 2021-05-05, AK: Bug fix for MLE error computation
# - 2021-05-04, AK: Bug fix for error computation
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
from computeMLEError import computeMLEError
from deadVolumeWrapper import deadVolumeWrapper
from extractDeadVolume import filesToProcess # File processing script
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

# File with parameter estimates
fileParameter = 'zlcParameters_20210820_0106_ea32ed7.npz'

# Flag to plot dead volume results
# Dead volume files have a certain name, use that to find what to plot
if fileParameter[0:10] == 'deadVolume':
    flagDeadVolume = True
else:
    flagDeadVolume = False

# Flag to plot simulations
simulateModel = True

# Flag to plot dead volume results
plotFt = False
    
# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);

# Plot colors
colorsForPlot = ["#faa307","#d00000","#03071e"]*4
markerForPlot = ["o"]*20

if flagDeadVolume:
    # Plot colors
    colorsForPlot = ["#FE7F2D","#B56938","#6C5342","#233D4D"]
    # File name of the experiments
    rawFileName = ['ZLC_DeadVolume_Exp23A_Output.mat',
                   'ZLC_DeadVolume_Exp23B_Output.mat',
                   'ZLC_DeadVolume_Exp23C_Output.mat',
                   'ZLC_DeadVolume_Exp23D_Output.mat',]

    # Dead volume parameter model path
    parameterPath = os.path.join('..','simulationResults',fileParameter)
   
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'DV')
    # Get the processed file names
    fileName = filesToProcess(False,[],[],'DV')
    # Load file names and the model
    fileNameList = load(parameterPath, allow_pickle=True)["fileName"]
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    x = modelOutputTemp[()]["variable"]

    # This was added on 12.06 (not back compatible for error computation)
    downsampleData = load(parameterPath)["downsampleFlag"]

    # Get the MS fit flag, flow rates and msDeadVolumeFile (if needed)
    # Check needs to be done to see if MS file available or not
    # Checked using flagMSDeadVolume in the saved file
    dvFileLoadTemp = load(parameterPath)
    if 'flagMSDeadVolume' in dvFileLoadTemp.files:
        flagMSFit = dvFileLoadTemp["flagMSFit"]
        msFlowRate = dvFileLoadTemp["msFlowRate"]
        flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
        msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
    else:
        flagMSFit = False
        msFlowRate = -np.inf
        flagMSDeadVolume = False
        msDeadVolumeFile = []
    
    numPointsExp = np.zeros(len(fileName))
    for ii in range(len(fileName)): 
        fileToLoad = fileName[ii]
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
    
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)

    # Print the objective function and volume from model parameters
    print("Objective Function",round(modelOutputTemp[()]["function"],0))
    print("Model Volume",round(sum(x[0:2]),2))
    computedError = 0
    numPoints = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])

    # Create the instance for the plots
    fig = plt.figure
    ax1 = plt.subplot(1,2,1)        
    ax2 = plt.subplot(1,2,2)
    # Initialize error for objective function
    # Loop over all available files    
    for ii in range(len(fileName)):
        # Initialize outputs
        moleFracSim = []
        # Path of the file name
        fileToLoad = fileName[ii]   
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
        flowRateTemp = load(fileToLoad)["flowRate"].flatten()
        timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
        moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
        flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
        # Get the flow rates from the fit file
        # When MS used
        if flagMSFit:
            flowRateDV = msFlowRate
        else:
            flowRateDV = np.mean(flowRateExp[-1:-10:-1])
        
        # Integration and ode evaluation time
        timeInt = timeElapsedExp
        
        # Print experimental volume 
        print("Experiment",str(ii+1),round(np.trapz(moleFracExp,
                                                    np.multiply(flowRateExp,timeElapsedExp)),2))
        if simulateModel:    
            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            moleFracSim = deadVolumeWrapper(timeInt, flowRateDV, x, flagMSDeadVolume, msDeadVolumeFile)
        
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(moleFracSim,
                                                      np.multiply(flowRateExp,
                                                                  timeElapsedExp)),2))
            
            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - minExp) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))
            
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
                    xlim = [0,150], ylim = [0, 1])   
            ax1.locator_params(axis="x", nbins=5)
            ax1.locator_params(axis="y", nbins=5)
            ax1.legend()
    
            # Log scale
            ax2.semilogy(timeElapsedExp,moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$t$ [s]', 
                    xlim = [0,150], ylim = [1e-2, 1])   
            ax2.locator_params(axis="x", nbins=5)

            
            #  Save the figure
            if saveFlag:
                # FileName: deadVolumeCharacteristics_<currentDateTime>_<GitCommitID_Current>_<modelFile>
                saveFileName = "deadVolumeCharacteristics_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
                savePath = os.path.join('..','simulationFigures',saveFileName)
                # Check if simulationFigures directory exists or not. If not, create the folder
                if not os.path.exists(os.path.join('..','simulationFigures')):
                    os.mkdir(os.path.join('..','simulationFigures'))
                plt.savefig (savePath)
        else:
            # Linear scale
            ax1.plot(np.multiply(flowRateDV,timeElapsedExp),moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.05,label=str(round(np.mean(flowRateDV),2))+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(np.multiply(flowRateDV,timeElapsedExp),moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$Ft$ [cc]', 
                    ylabel='$y_1$ [-]',
                    xlim = [0,0.1], ylim = [0, 1])         
            ax1.legend()
    
            # Log scale
            ax2.semilogy(np.multiply(flowRateDV,timeElapsedExp),moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.05,label=str(round(np.mean(flowRateDV),2))+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(np.multiply(flowRateDV,timeElapsedExp),moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$Ft$ [cc]', 
                    xlim = [0,0.1], ylim = [1e-3, 1])         
            ax2.legend()
            
            #  Save the figure
            if saveFlag:
                # FileName: deadVolumeCharacteristicsFt_<currentDateTime>_<GitCommitID_Current>_<modelFile>
                saveFileName = "deadVolumeCharacteristicsFt_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
                savePath = os.path.join('..','simulationFigures',saveFileName)
                # Check if simulationFigures directory exists or not. If not, create the folder
                if not os.path.exists(os.path.join('..','simulationFigures')):
                    os.mkdir(os.path.join('..','simulationFigures'))
                plt.savefig (savePath)
    plt.show()
    # Print the MLE error
    if simulateModel:
        computedError = computeMLEError(moleFracExpALL,moleFracSimALL,
                                        downsampleData=downsampleData,)
        print("Sanity check objective function: ",round(computedError,0))
    
    # Remove all the .npy files genereated from the .mat
    # Loop over all available files    
    for ii in range(len(fileName)):
        os.remove(fileName[ii])

else:
    from simulateCombinedModel import simulateCombinedModel

    # File name of the experiments
    rawFileName = ['ZLC_ActivatedCarbon_Exp72A_Output.mat',
                    'ZLC_ActivatedCarbon_Exp74A_Output.mat',
                    'ZLC_ActivatedCarbon_Exp76A_Output.mat',
                    'ZLC_ActivatedCarbon_Exp72B_Output.mat',
                    'ZLC_ActivatedCarbon_Exp74B_Output.mat',
                    'ZLC_ActivatedCarbon_Exp76B_Output.mat',
                    'ZLC_ActivatedCarbon_Exp73A_Output.mat',
                    'ZLC_ActivatedCarbon_Exp75A_Output.mat',
                    'ZLC_ActivatedCarbon_Exp77A_Output.mat',
                    'ZLC_ActivatedCarbon_Exp73B_Output.mat',
                    'ZLC_ActivatedCarbon_Exp75B_Output.mat',
                    'ZLC_ActivatedCarbon_Exp77B_Output.mat',]

    # ZLC parameter model path
    parameterPath = os.path.join('..','simulationResults',fileParameter)
    
    # Temperature (for each experiment)
    temperatureExp = [344.69, 325.39, 306.15]*4
    # Legend flag
    useFlow = False
    
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
    # Get the processed file names
    fileName = filesToProcess(False,[],[],'ZLC')
    # Mass of sorbent and particle epsilon
    adsorbentDensity = load(parameterPath)["adsorbentDensity"]
    particleEpsilon = load(parameterPath)["particleEpsilon"]
    massSorbent = load(parameterPath)["massSorbent"]
    # Volume of sorbent material [m3]
    volSorbent = (massSorbent/1000)/adsorbentDensity
    
    # Volume of gas chamber (dead volume) [m3]
    volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

    # Dead volume model
    deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
    # Isotherm parameter reference
    parameterReference = load(parameterPath)["parameterReference"]
    # Load the model
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    modelNonDim = modelOutputTemp[()]["variable"] 

    # This was added on 12.06 (not back compatible for error computation)
    downsampleData = load(parameterPath)["downsampleFlag"]
    print("Objective Function",round(modelOutputTemp[()]["function"],0))

    numPointsExp = np.zeros(len(fileName))
    for ii in range(len(fileName)): 
        fileToLoad = fileName[ii]
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
    
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)
    # Multiply the paremeters by the reference values
    x = np.multiply(modelNonDim,parameterReference)
    # Initialize loadings
    computedError = 0
    numPoints = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])
    massBalanceALL = np.zeros((len(fileName),3))

    # Create the instance for the plots
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)        
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
                
    # Loop over all available files    
    for ii in range(len(fileName)):
        fileToLoad = fileName[ii]   
        
        # Initialize outputs
        moleFracSim = []  
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
        flowRateTemp = load(fileToLoad)["flowRate"].flatten()
        timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
        moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
        flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                
        # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
        timeInt = timeElapsedExp
        
        # Print experimental volume 
        print("Experiment",str(ii+1),round(np.trapz(np.multiply(flowRateExp,moleFracExp),timeElapsedExp),2))

        if simulateModel:
            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                        flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                        expFlag = True,
                                                        isothermModel = isothermModel,
                                                        rateConstant_1 = rateConstant_1,
                                                        rateConstant_2 = rateConstant_2,
                                                        deadVolumeFile = deadVolumeFile,
                                                        volSorbent = volSorbent,
                                                        volGas = volGas,
                                                        temperature = temperatureExp[ii],
                                                        adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Compute the mass balance at the end end of the ZLC
            massBalanceALL[ii,0] = moleFracExp[0]
            massBalanceALL[ii,1] = ((np.trapz(np.multiply(resultMat[3,:],resultMat[0,:]),timeElapsedExp)
                                    - volGas*moleFracExp[0])*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000))
            massBalanceALL[ii,2] = volGas*moleFracExp[0]*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000)   

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
            
        # y - Linear scale
        ax1.semilogy(timeElapsedExp,moleFracExp,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.1) # Experimental response
        if simulateModel:
            if useFlow:
                legendStr = str(round(np.mean(flowRateExp),2))+" ccs"
            else:
                legendStr = str(temperatureExp[ii])+" K"
            ax1.plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],label=legendStr) # Simulation response    
            # if ii==len(fileName)-1:
            #     ax1.plot(timeElapsedExp,moleFracDV,
            #              color='#118ab2',label="DV",alpha=0.025,
            #              linestyle = '-') # Dead volume simulation response    

        ax1.set(xlabel='$t$ [s]', 
                ylabel='$y_1$ [-]',
                xlim = [0,200], ylim = [1e-2, 1])    
        ax1.locator_params(axis="x", nbins=4)
        # ax1.legend()

        # Ft - Log scale        
        ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracExp,
                      marker = markerForPlot[ii],linewidth = 0,
                      color=colorsForPlot[ii],alpha=0.1) # Experimental response
        if simulateModel:
            ax2.semilogy(np.multiply(resultMat[3,:]*1e6,timeElapsedExp),moleFracSim,
                          color=colorsForPlot[ii],label=str(round(np.mean(resultMat[3,:]*1e6),2))+" ccs") # Simulation response
        ax2.set(xlabel='$Ft$ [cc]', 
                xlim = [0,60], ylim = [1e-2, 1])         
        ax2.locator_params(axis="x", nbins=4)
        
        # Flow rates
        ax3.plot(timeElapsedExp,flowRateExp,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.1,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
        if simulateModel:
            ax3.plot(timeElapsedExp,resultMat[3,:]*1e6,
                      color=colorsForPlot[ii]) # Simulation response    
        ax3.set(xlabel='$t$ [s]', 
                ylabel='$F$ [ccs]',
                xlim = [0,250], ylim = [0, 1.5])
        ax3.locator_params(axis="x", nbins=4)
        ax3.locator_params(axis="y", nbins=4)

        #  Save the figure
        if saveFlag:
            # FileName: zlcCharacteristics_<currentDateTime>_<GitCommitID_Current>_<modelFile>
            saveFileName = "zlcCharacteristics_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
            savePath = os.path.join('..','simulationFigures',saveFileName)
            # Check if simulationFigures directory exists or not. If not, create the folder
            if not os.path.exists(os.path.join('..','simulationFigures')):
                os.mkdir(os.path.join('..','simulationFigures'))
            plt.savefig (savePath)         
        
    plt.show()
    
    # Print the MLE error
    if simulateModel:
        computedError = computeMLEError(moleFracExpALL,moleFracSimALL, 
                                        downsampleData = downsampleData,)
        print("Sanity check objective function: ",round(computedError,0))
    
    # Remove all the .npy files genereated from the .mat
    # Loop over all available files    
    for ii in range(len(fileName)):
        os.remove(fileName[ii])