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
from simulateDeadVolume import simulateDeadVolume
from computeMLEError import computeMLEError
from computeEquilibriumLoading import computeEquilibriumLoading
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
flagDeadVolume = False

# Flag to plot simulations
simulateModel = True

# Flag to plot dead volume results
plotFt = False

# Adsorbent density [kg/m3]
# This has to be the skeletal density
adsorbentDensity = 1950 # Activated carbon skeletal density [kg/m3]

# Particle porosity
particleEpsilon = 0.61

# Particle mass [g]
massSorbent = 0.0846

# Volume of sorbent material [m3]
volSorbent = (massSorbent/1000)/adsorbentDensity

# Volume of gas chamber (dead volume) [m3]
volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
    
# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);
    
# Temperature of the gas [K]
temperature = np.array([298.15]);
    

# Directory of raw data
mainDir = '/Users/ash23win/Google Drive/ERASE/experimental/runData/'

colorsForPlot = ["#E5383B","#E5383B","#B55055","#B55055","#6C757D","#6C757D"]
colorsForPlot = ["#E5383B","#CD4448","#B55055",
                 "#9C5D63","#846970","#6C757D"]
# colorsForPlot = ["#E5383B","#6C757D"]
markerForPlot = ["o","v","o","v","o","v"]

if flagDeadVolume:
    # File name of the experiments
    fileName = ['ZLC_DeadVolume_Exp16B_Output_ba091f5.npz',
                'ZLC_DeadVolume_Exp16C_Output_ba091f5.npz',
                'ZLC_DeadVolume_Exp16D_Output_ba091f5.npz']
    # File with parameter estimates
    simulationDir = '/Users/ash23win/Google Drive/ERASE/simulationResults/'
    fileParameter = 'deadVolumeCharacteristics_20210511_1203_ebb447e.npz'
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
    computedError = 0
    numPoints = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])

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
            
            # Stack mole fraction from experiments and simulation
            moleFracExpALL = np.hstack((moleFracExpALL, moleFracExp))
            moleFracSimALL = np.hstack((moleFracSimALL, moleFracSim))

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
                    xlim = [0,100], ylim = [0, 1])         
            ax1.legend()
    
            # Log scale
            ax2.semilogy(timeElapsedExp,moleFracExp,
                          marker = markerForPlot[ii],linewidth = 0,
                          color=colorsForPlot[ii],alpha=0.2,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
            if simulateModel:
                ax2.semilogy(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response
            ax2.set(xlabel='$t$ [s]', 
                    xlim = [0,150], ylim = [5e-3, 1])         
            
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
       
    # Print the MLE error
    computedError = computeMLEError(moleFracExpALL,moleFracSimALL)
    print(round(computedError,0))
    
else:
    from simulateCombinedModel import simulateCombinedModel
    
    # Directory of raw data
    mainDir = '/Users/ash23win/Google Drive/ERASE/experimental/runData/'
    # File name of the experiments
    fileName = [
                'ZLC_ActivatedCarbon_Exp24A_Output_ba091f5.npz',
                'ZLC_ActivatedCarbon_Exp24B_Output_ba091f5.npz',
                'ZLC_ActivatedCarbon_Exp24C_Output_ba091f5.npz',
                'ZLC_ActivatedCarbon_Exp24D_Output_ba091f5.npz',
                'ZLC_ActivatedCarbon_Exp24E_Output_ba091f5.npz',]
    
    # File with parameter estimates
    simulationDir = '/Users/ash23win/Google Drive/ERASE/simulationResults/'
    # Dead volume model
    deadVolumeFile = 'deadVolumeCharacteristics_20210511_1015_ebb447e.npz'  
    # ZLC parameter model
    fileParameter = 'zlcParameters_20210513_2239_ba091f5.npz'
    modelOutputTemp = load(simulationDir+fileParameter, allow_pickle=True)["modelOutput"]
    print("Objective Function",round(modelOutputTemp[()]["function"],0))
    modelNonDim = modelOutputTemp[()]["variable"] 
    numPointsExp = np.zeros(len(fileName))
    for ii in range(len(fileName)): 
        fileToLoad = mainDir + fileName[ii]
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
    
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)

    # Multiply the paremeters by the reference values (for SSL)
    x = np.multiply(modelNonDim,[10, 1e-5, 50e3,100])

    # Ronny AC Data
    x_RP = [0.44, 3.17e-6, 28.63e3, 6.10, 3.21e-6, 20.37e3,100]

    # Initialize loadings
    computedError = 0
    numPoints = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])
    massBalanceALL = np.zeros((len(fileName),2))

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
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                        flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                        expFlag = True,
                                                        isothermModel=x[0:-1],
                                                        rateConstant=x[-1],
                                                        deadVolumeFile = deadVolumeFile,
                                                        volSorbent = volSorbent,
                                                        volGas = volGas)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation
            moleFracExpALL = np.hstack((moleFracExpALL, moleFracExp))
            moleFracSimALL = np.hstack((moleFracSimALL, moleFracSim))
        
            # Compute the mass balance at the end end of the ZLC
            massBalanceALL[ii,0] = moleFracExp[0]
            massBalanceALL[ii,1] = ((np.trapz(np.multiply(resultMat[3,:],resultMat[0,:]),timeElapsedExp)
                                    - volGas*moleFracExp[0])*(pressureTotal/(8.314*temperature))/(massSorbent/1000))
        
        # y - Linear scale
        ax1.semilogy(timeElapsedExp,moleFracExp,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.25) # Experimental response
        if simulateModel:
            ax1.plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],label=str(round(np.mean(flowRateExp),2))+" ccs") # Simulation response    
        ax1.set(xlabel='$t$ [s]', 
                ylabel='$y_1$ [-]',
                xlim = [0,300], ylim = [5e-3, 1])    
        ax1.locator_params(axis="x", nbins=4)
        ax1.legend()
    
        # Ft - Log scale        
        ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracExp,
                      marker = markerForPlot[ii],linewidth = 0,
                      color=colorsForPlot[ii],alpha=0.25) # Experimental response
        if simulateModel:
            ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracSim,
                          color=colorsForPlot[ii],label=str(round(np.mean(flowRateExp),2))+" ccs") # Simulation response
        ax2.set(xlabel='$Ft$ [cc]', 
                xlim = [0,100], ylim = [5e-3, 1])         
        ax2.locator_params(axis="x", nbins=4)
        
        # Flow rates
        ax3.plot(timeElapsedExp,flowRateExp,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.25,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
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
    
    # Print the MLE error
    computedError = computeMLEError(moleFracExpALL,moleFracSimALL)
    print(round(computedError,0))
    
    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)
    # Initialize isotherms 
    isoLoading_RP = np.zeros([len(y)])
    isoLoading_ZLC = np.zeros([len(y)])

    # Loop over all the mole fractions
    for ii in range(len(y)):
        isoLoading_RP[ii] = computeEquilibriumLoading(isothermModel=x_RP[0:-1],
                                                      moleFrac = y[ii])
        isoLoading_ZLC[ii] = computeEquilibriumLoading(isothermModel=x[0:-1],
                                                        moleFrac = y[ii])

    # Plot the isotherms   
    os.chdir(os.path.join('..','plotFunctions'))
    plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure
    ax1 = plt.subplot(1,1,1)        
    ax1.plot(y,isoLoading_RP,color='#2a9d8f',label="Autosorb") # Ronny's isotherm
    ax1.plot(y,isoLoading_ZLC,color='#e76f51',label="ZLC") # ALL
    ax1.scatter(massBalanceALL[:,0],massBalanceALL[:,1],c='dimgrey')
    ax1.set(xlabel='$P$ [bar]', 
            ylabel='$q^*$ [mol kg$^\mathregular{-1}$]',
            xlim = [0,1], ylim = [0, 3]) 
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)        
    ax1.legend()
    #  Save the figure
    if saveFlag:
        # FileName: isothermComparison_<currentDateTime>_<GitCommitID_Current>_<modelFile>
        saveFileName = "isothermComparison_" + currentDT + "_" + gitCommitID + "_" + fileParameter[-25:-12] + saveFileExtension
        savePath = os.path.join('..','simulationFigures',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)         
    
    plt.show()