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
import pdb
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

## DIFFUSION FITS const dEFF const kEFF LOW MASS mg LOW comp 100 GP ISOTHERMAL TAU FIT PARALLEL PORE Vcell DV
fileParameter = 'zlcParameters_ZYHCrush_20240518_2113_b571c46.npz' # ZYH ALL FLOW Diff
fileParameter = 'zlcParameters_ZYNaCrush_20240521_2240_b571c46.npz' # ZYNa ALL FLOW Diff
fileParameter = 'zlcParameters_ZYTMACrush_20240520_1927_b571c46.npz' # ZYTMA ALL FLOW Diff

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
# colorsForPlot =['r','g','b']*4

markerForPlot = ["o"]*20

if flagDeadVolume:
    # Plot colors
    colorsForPlot = ["#FE7F2D","#B56938","#6C5342","#233D4D"]*2
    # File name of the experiments
    # Dead volume parameter model path
    parameterPath = os.path.join('..','simulationResults',fileParameter)
    fileNameList = load(parameterPath, allow_pickle=True)["fileName"]
    rawFileName = fileNameList

    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'DV')
    # Get the processed file names
    fileName = filesToProcess(False,[],[],'DV')
    
    # Load file names and the model
    
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    x = modelOutputTemp[()]["variable"]

    # This was added on 12.06 (not back compatible for error computation)
    downsampleData = load(parameterPath)["downsampleFlag"]
    downsampleData = True
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
            moleFracSim = deadVolumeWrapper(timeInt, flowRateDV, x, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = moleFracExp[0])
        
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
                          color=colorsForPlot[ii],alpha=0.3,label=str(round(np.mean(flowRateExp[-1:-10:-1]),2))+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$t$ [s]', 
                    ylabel='$y_1$ [-]',
                    ylim =  [0, 1])   
            ax1.autoscale(enable=None, axis="x", tight=False)  
            ax1.locator_params(axis="x", nbins=5)
            ax1.locator_params(axis="y", nbins=5)
            ax1.legend()
    
            # Log scale
            ax2.scatter(timeElapsedExp[0::1],moleFracExp[0::1],
                          marker = markerForPlot[ii],
                          color='none', edgecolors = colorsForPlot[ii],alpha=0.3,label=str(round(flowRateExp[-1],2))+" ccs", s =20, linewidth = 1, linestyle = '-') # Experimental response
            if simulateModel:
                ax2.semilogy(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii], linewidth = 1) # Simulation response
            ax2.set(xlabel='$t$ [s]', 
                    ylim =  [1e-3, 1])   
            ax2.autoscale(enable=None, axis="x", tight=False)  
            ax2.locator_params(axis="x", nbins=5)
            ax2.legend()

            
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
                          color=colorsForPlot[ii],alpha=0.3,label=str(round(np.mean(flowRateDV),2))+" ccs") # Experimental response
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
                          color=colorsForPlot[ii],alpha=0.3,label=str(round(np.mean(flowRateDV),2))+" ccs") # Experimental response
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
        
    # ZLC parameter model path
    parameterPath = os.path.join('..','simulationResults',fileParameter)  

    # Legend flag
    useFlow = False
    
    rawFileName = load(parameterPath)["fileName"]
    
    temperatureExp = load(parameterPath)["temperature"]

    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
    # Get the processed file names
    fileName = filesToProcess(False,[],[],'ZLC')
    # Mass of sorbent and particle epsilon
    adsorbentDensity = load(parameterPath)["adsorbentDensity"]
    particleEpsilon = load(parameterPath)["particleEpsilon"]
    massSorbent = load(parameterPath)["massSorbent"]
    deadVolumeFile = load(parameterPath)["deadVolumeFile"]
 
    # Volume of sorbent material [m3]
    volSorbent = (massSorbent/1000)/adsorbentDensity
    
    # Volume of gas chamber (dead volume) [m3]
    volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

    # Dead volume model
    deadVolumeFile = load(parameterPath)["deadVolumeFile"]

    # Isotherm parameter reference
    parameterReference = load(parameterPath)["parameterReference"]
    paramIso = load(parameterPath)["paramIso"]
    # Load the model
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    modelNonDim = modelOutputTemp[()]["variable"] 

    # This was added on 12.06 (not back compatible for error computation)
    downsampleData = load(parameterPath)["downsampleFlag"]

    modelType = load(parameterPath)["modelType"]
    if modelType == 'Diffusion1Ttau' or modelType == 'Diffusion1TNItau':
        # mean pore radius from MIP
        rpore = load(parameterPath)["rpore"]
        Dpvals = load(parameterPath)["Dpvals"]
        numPellets = load(parameterPath)["numPellets"]
    else:
        rpore = 107e-9
        Dpvals = [3.05671321526166e-05,	3.15050794527196e-05,	3.24331710687508e-05]
        numPellets = 3

    print("Objective Function",round(modelOutputTemp[()]["function"],0))

    numPointsExp = np.zeros(len(fileName))
    for ii in range(len(fileName)): 
        fileToLoad = fileName[ii]
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
    
    # Downsample intervals
    downsampleInt = numPointsExp/numPointsExp
        
    # Multiply the paremeters by the reference values
    x = np.multiply(modelNonDim,parameterReference)
    
    if len(paramIso) <10:
        if modelType == 'KineticSBMacro':
            x = np.zeros(9)
            x[0:6] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)
        if modelType == 'KineticSBMacro2':
            x = np.zeros(9)
            x[0:6] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)    
        elif modelType == 'Diffusion':
            x = np.zeros(9)
            x[0:6] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'Diffusion1T':
            x = np.zeros(8)
            x[0:6] = paramIso[0:-2]
            x[-2:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'Diffusion1Ttau':
            x = np.zeros(8)
            x[0:6] = paramIso[0:-2]
            x[-2:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'Diffusion1TNI':
            x = np.zeros(8)
            x[0:6] = paramIso[0:-2]
            x[-2:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'KineticMacro':
            x = np.zeros(9)
            x[0:6] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)      
        else:
            x = np.zeros(8)
            x[0:6] = paramIso[0:-2]
            x[-2:] = np.multiply(modelNonDim,parameterReference)    
    elif len(paramIso) == 13:
        if modelType == 'KineticSBMacro':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)
        if modelType == 'KineticSBMacro2':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)    
        elif modelType == 'Diffusion':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'Diffusion1T':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-2:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'Diffusion1TNI':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-2:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'Diffusion1TNItau':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-2:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'Diffusion1Ttau':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-2:] = np.multiply(modelNonDim,parameterReference)
        elif modelType == 'KineticMacro':
            x = np.zeros(13)
            x[0:10] = paramIso[0:-3]
            x[-3:] = np.multiply(modelNonDim,parameterReference)      
        else:
            x = np.zeros(12)
            x[0:10] = paramIso[0:-2]
            x[-2:] = np.multiply(modelNonDim,parameterReference) 
    
    computedError = 0
    numPoints = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])
    timeElapsedExpFULL = np.array([])
    moleFracExpFULL = np.array([])
    moleFracSimFULL = np.array([])
    moleFracZLCFULL = np.array([])
    moleFracDVFULL = np.array([])
    qAverageFULL = np.array([])
    flowRateSimFULL = np.array([])
    flowRateExpFULL = np.array([])
    massBalanceALL = np.zeros((len(fileName),3))
    massBalanceEXP = np.zeros((len(fileName),3))
    massBalanceDV = np.zeros((len(fileName),3))

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
        if moleFracExp[0] > 0.5:
            deadVolumeFlow = deadVolumeFile[1]
        else:
            deadVolumeFlow = deadVolumeFile[0]
            
        if len(deadVolumeFlow[0]) == 1: # 1 DV for 1 DV file
            deadVolumeFileTemp = str(deadVolumeFlow[0])
        else:
            if np.absolute(flowRateExp[-1] - 1) > 0.2: # for lowflowrate experiments!
                deadVolumeFileTemp =  str(deadVolumeFlow[0][0])
            else:
                deadVolumeFileTemp =  str(deadVolumeFlow[0][1])  
                
        # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
        timeInt = timeElapsedExp

        # Print experimental volume 
        print("Experiment",str(ii+1),round(np.trapz(moleFracExp,np.multiply(flowRateExp,timeElapsedExp)),2))
        
        # modelType = 'Diffusion'
        
        if simulateModel:
            # Parse out parameter values
            if modelType == 'KineticSBMacro':
                isothermModel = x[0:-3]
                rateConstant_1 = x[-3]
                rateConstant_2 = x[-2]
                rateConstant_3 = x[-1]     
            if modelType == 'KineticSBMacro2':
                isothermModel = x[0:-3]
                rateConstant_1 = x[-3]
                rateConstant_2 = x[-2]
                rateConstant_3 = x[-1]     
            elif modelType == 'KineticMacro':
                isothermModel = x[0:-3]
                rateConstant_1 = x[-3]
                rateConstant_2 = x[-2]
                rateConstant_3 = x[-1]  
            elif modelType == 'Diffusion':
                isothermModel = x[0:-3]
                rateConstant_1 = x[-3]
                rateConstant_2 = x[-2]
                rateConstant_3 = x[-1]              
            elif modelType == 'Diffusion1T':
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = 0
                rateConstant_3 = x[-1]  
            elif modelType == 'Diffusion1TNI':
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = 0
                rateConstant_3 = x[-1]
            elif modelType == 'Diffusion1TNItau':
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = 0
                rateConstant_3 = x[-1]
            elif modelType == 'Diffusion1Ttau':
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = 0
                rateConstant_3 = x[-1]
            else:
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = x[-1]
                rateConstant_3 = 0        
                
            # Compute the combined zlc and dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                        flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                        expFlag = True,
                                                        isothermModel = isothermModel,
                                                        rateConstant_1 = rateConstant_1,
                                                        rateConstant_2 = rateConstant_2,
                                                        rateConstant_3 = rateConstant_3,
                                                        deadVolumeFile = deadVolumeFileTemp,
                                                        volSorbent = volSorbent,
                                                        volGas = volGas,
                                                        rpore = rpore,
                                                        numPellets = numPellets,
                                                        Dpvals = Dpvals,
                                                        temperature = temperatureExp[ii],
                                                        adsorbentDensity = adsorbentDensity,
                                                        modelType = modelType)

            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),timeInt),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))
            
            timeElapsedExpFULL  = np.hstack((timeElapsedExpFULL, timeInt))
            moleFracSimFULL  = np.hstack((moleFracSimFULL, moleFracSim))
            moleFracZLCFULL  = np.hstack((moleFracZLCFULL, np.transpose(resultMat[0,:])))
            moleFracExpFULL  = np.hstack((moleFracExpFULL, moleFracExp))
            qAverageFULL = np.hstack((qAverageFULL, np.transpose(resultMat[1,:])))
            flowRateSimFULL = np.hstack((flowRateSimFULL, np.transpose(resultMat[3,:])))
            flowRateExpFULL = np.hstack((flowRateExpFULL,flowRateExp/1e6))
            
            # # Compute the mass balance at the end of the ZLC
            # massBalanceALL[ii,0] = moleFracExp[0]
            # massBalanceALL[ii,1] = ((np.trapz(resultMat[0,:],np.multiply(resultMat[3,:],timeInt))
            #                         - volGas*moleFracExp[0])*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000))
            # massBalanceALL[ii,2] = volGas*moleFracExp[0]*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000)   
        
            flowInDV = np.zeros((len(resultMat[1,:])))          
            flowInDV[:] = np.mean(flowRateExp[-1:-2:-1]*1e-6)
            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFileTemp)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, flowInDV*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
            moleFracDVFULL  = np.hstack((moleFracDVFULL, moleFracDV))

        # y - Linear scale
        ax1.semilogy(timeElapsedExp[0::10],moleFracExp[0::10], 
                marker = markerForPlot[ii],linewidth = 0,
                      alpha=0.8,markeredgecolor=colorsForPlot[ii],markeredgewidth=0.1,markerfacecolor='none') # Experimental response
        if simulateModel:
            if useFlow:
                legendStr = str(round(np.mean(flowRateExp),2))+" ccs"
            else:
                legendStr = str(temperatureExp[ii])+" K"
            ax1.plot(timeInt,moleFracSim,
                     color=colorsForPlot[ii],label=legendStr,alpha = 1) # Simulation response    
            # if ii==len(fileName)-1:
            ax1.plot(timeInt,moleFracDV,
                          color='#118ab2',label="DV",alpha=0.1,
                          linestyle = '-') # Dead volume simulation response    

        ax1.set(xlabel='$t$ [s]', 
                ylabel='$y_1$ [-]',
                ylim =  [8e-3, 1])
                # xlim =  [0, 100])   
        ax1.locator_params(axis="x", nbins=4)
        ax1.autoscale(enable=None, axis="x", tight=False)

        # ax1.legend()

        # Ft - Log scale        
        ax2.semilogy(np.multiply(flowRateExp[0::10],timeElapsedExp[0::10]),moleFracExp[0::10],
                      marker = markerForPlot[ii],linewidth = 0,
                      alpha=0.8,markeredgecolor=colorsForPlot[ii],markeredgewidth=0.1,markerfacecolor='none') # Experimental response
        if simulateModel:
            ax2.semilogy(np.multiply(resultMat[3,:]*1e6,timeInt),moleFracSim,
                          color=colorsForPlot[ii],label=str(round(np.mean(resultMat[3,:]*1e6),2))+" ccs",
                          alpha = 1) # Simulation response
            ax2.plot(np.multiply(timeInt,flowInDV*1e6),moleFracDV,
                          color='#118ab2',label="DV",alpha=0.1,
                          linestyle = '-') # Dead volume simulation response  
        ax2.set(xlabel='$Ft$ [cc]', 
                ylim =  [8e-3, 1.1*np.max(moleFracSim)])   
        ax2.locator_params(axis="x", nbins=4)
        ax2.locator_params(axis="x", nbins=4)
        ax2.autoscale(enable=None, axis="x", tight=False)

        # Flow rates
        ax3.plot(timeElapsedExp,flowRateExp,
                marker = markerForPlot[ii],linewidth = 0,
                alpha=0.8,markeredgecolor=colorsForPlot[ii],markeredgewidth=0.1,markerfacecolor='none',label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
        if simulateModel:
            ax3.plot(timeInt,resultMat[3,:]*1e6,
                      color=colorsForPlot[ii]) # Simulation response    
        ax3.set(xlabel='$t$ [s]', 
                ylabel='$F$ [ccs]',
                xlim = [0,100], ylim = [0, 3])
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
        # pdb.set_trace() 
    plt.show()
    
    # Print the MLE error
    if simulateModel:
        computedError = computeMLEError(moleFracExpALL,moleFracSimALL, 
                                        downsampleData = downsampleData,)
        print("Sanity check objective function: ",round(computedError,4))
    
    # fig = plt.figure
    # ax1 = plt.subplot(1,2,1)        
    # ax2 = plt.subplot(1,2,2)
    
    # # Loop over all available files    
    # for ii in range(len(fileName)):
    #     fileToLoad = fileName[ii]   
        
    #     # Initialize outputs
    #     moleFracSim = []  
    #     # Load experimental time, molefraction and flowrate (accounting for downsampling)
    #     timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
    #     moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
    #     flowRateTemp = load(fileToLoad)["flowRate"].flatten()
    #     timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
    #     # if ii < 3:
    #     #     timeElapsedExp = timeElapsedExp-0.2
    #     # timeElapsedExp[timeElapsedExp < 0] = 0
    #     moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
    #     flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
    #     if moleFracExp[0] > 0.5:
    #         deadVolumeFlow = deadVolumeFile[1]
    #     else:
    #         deadVolumeFlow = deadVolumeFile[0]
            
    #     if len(deadVolumeFlow[0]) == 1: # 1 DV for 1 DV file
    #         deadVolumeFileTemp = str(deadVolumeFlow[0])
    #     else:
    #         if np.absolute(flowRateExp[-1] - 1) > 0.2: # for lowflowrate experiments!
    #             deadVolumeFileTemp =  str(deadVolumeFlow[0][0])
    #         else:
    #             deadVolumeFileTemp =  str(deadVolumeFlow[0][1])  
                
    #     # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
    #     timeInt = timeElapsedExp
        
    #     # Print experimental volume 
    #     print("Experiment",str(ii+1),round(np.trapz(moleFracExp,np.multiply(flowRateExp,timeElapsedExp)),2))

    #     if simulateModel:
    #         # Parse out parameter values
    #         if modelType == 'KineticSBMacro':
    #             isothermModel = x[0:-3]
    #             rateConstant_1 = x[-3]
    #             rateConstant_2 = x[-2]
    #             rateConstant_3 = x[-1]     
    #         elif modelType == 'KineticMacro':
    #             isothermModel = x[0:-3]
    #             rateConstant_1 = x[-3]
    #             rateConstant_2 = x[-2]
    #             rateConstant_3 = x[-1]  
    #         else:
    #             isothermModel = x[0:-2]
    #             rateConstant_1 = x[-2]
    #             rateConstant_2 = x[-1]
    #             rateConstant_3 = 0        
    #         # Compute the combined zlc and dead volume response using the optimizer parameters
    #         _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
    #                                                     initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
    #                                                     flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
    #                                                     expFlag = True,
    #                                                     isothermModel = isothermModel,
    #                                                     rateConstant_1 = rateConstant_1,
    #                                                     rateConstant_2 = rateConstant_2,
    #                                                     rateConstant_3 = rateConstant_3,
    #                                                     deadVolumeFile = deadVolumeFileTemp,
    #                                                     volSorbent = volSorbent,
    #                                                     volGas = volGas,
    #                                                     temperature = temperatureExp[ii],
    #                                                     adsorbentDensity = adsorbentDensity,
    #                                                     modelType = modelType)
    #         # Print simulation volume    
    #         print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
    #                                                               moleFracSim),
    #                                                     timeElapsedExp),2))

    #         # Stack mole fraction from experiments and simulation for error 
    #         # computation
    #         minExp = np.min(moleFracExp) # Compute the minimum from experiment
    #         normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
    #         moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
    #         moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

    #         # Compute the mass balance at the end of the ZLC
    #         massBalanceALL[ii,0] = moleFracExp[0]
    #         massBalanceALL[ii,1] = ((np.trapz(resultMat[0,:],np.multiply(resultMat[3,:],timeElapsedExp))
    #                                 - volGas*moleFracExp[0])*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000))
    #         massBalanceALL[ii,2] = volGas*moleFracExp[0]*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000)   
        
    #         flowInDV = np.zeros((len(resultMat[1,:])))          
    #         flowInDV[:] = np.mean(flowRateExp[-1:-2:-1]*1e-6)
    #         # Call the deadVolume Wrapper function to obtain the outlet mole fraction
    #         deadVolumePath = os.path.join('..','simulationResults',deadVolumeFileTemp)
    #         modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
    #         pDV = modelOutputTemp[()]["variable"]
    #         dvFileLoadTemp = load(deadVolumePath)
    #         flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
    #         msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
    #         moleFracDV = deadVolumeWrapper(timeInt, flowInDV*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
            
    #     # y - Linear scale
    #     if simulateModel:
    #         if useFlow:
    #             legendStr = str(round(np.mean(flowRateExp),2))+" ccs"
    #         else:
    #             legendStr = str(temperatureExp[ii])+" K"
    #         ax1.semilogy(timeElapsedExp,resultMat[0,:],
    #                   color=colorsForPlot[ii],label=legendStr,alpha = 1) # Simulation response 

    #     ax1.set(xlabel='$t$ [s]', 
    #             ylabel='$y_1$ [-]',
    #             ylim =  [1e-2, 1],
    #             xlim =  [0, 1000])   
    #     ax1.locator_params(axis="x", nbins=4)
    #     # ax1.autoscale(enable=None, axis="x", tight=False)

    #     # ax1.legend()

    #     # Ft - Log scale        
    #     if simulateModel:
    #         ax2.semilogy(np.multiply(resultMat[3,:]*1e6,timeElapsedExp),resultMat[0,:],
    #                       color=colorsForPlot[ii],label=str(round(np.mean(resultMat[3,:]*1e6),2))+" ccs",
    #                       alpha = 1) # Simulation response 
    #     ax2.set(xlabel='$Ft$ [cc]', 
    #             ylim =  [1e-2, 1])   
    #     ax1.locator_params(axis="x", nbins=4)

    #     ax2.locator_params(axis="x", nbins=4)
        
    #     # Loading
    #     if simulateModel:
    #         ax3.plot(timeElapsedExp,resultMat[1,:]/adsorbentDensity,
    #                   color=colorsForPlot[ii]) # Simulation response    
    #     ax3.set(xlabel='$t$ [s]', 
    #             ylabel='$q$ [mol/kg]',
    #             xlim = [0,100], ylim = [0, 3])
    #     ax3.locator_params(axis="x", nbins=4)
    #     ax3.locator_params(axis="y", nbins=4)

    #     #  Save the figure
    #     if saveFlag:
    #         # FileName: zlcCharacteristics_<currentDateTime>_<GitCommitID_Current>_<modelFile>
    #         saveFileName = "zlcCharacteristics_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
    #         savePath = os.path.join('..','simulationFigures',saveFileName)
    #         # Check if simulationFigures directory exists or not. If not, create the folder
    #         if not os.path.exists(os.path.join('..','simulationFigures')):
    #             os.mkdir(os.path.join('..','simulationFigures'))
    #         plt.savefig (savePath)        
    
           
    # plt.show()
    
    # # Print the MLE error
    # if simulateModel:
    #     computedError = computeMLEError(moleFracExpALL,moleFracSimALL, 
    #                                     downsampleData = downsampleData,)
    #     print("Sanity check objective function: ",round(computedError,0))
    
    # Print model data for sanity checks
    print("\nFurther Sanity Checks (from the parameter estimate file): ")
    print("Dead Volume File: ",str(deadVolumeFile))
    print("Adsorbent Density: ",str(adsorbentDensity)," kg/m3")
    print("Mass Sorbent: ",str(massSorbent)," g")
    print("Particle Porosity: ",str(particleEpsilon))
    print("File name list: ",load(parameterPath)["fileName"])
    print("Temperature: ",load(parameterPath)["temperature"])
    
    # Remove all the .npy files genereated from the .mat
    # Loop over all available files    
    for ii in range(len(fileName)):
        os.remove(fileName[ii])