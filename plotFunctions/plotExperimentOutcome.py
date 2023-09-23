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
fileParameter = 'zlcParameters_20220510_0309_e81a19e.npz' # AC SIM
# fileParameter = 'zlcParameters_20210823_1731_c8173b1.npz' # BN SIM
# fileParameter = 'zlcParameters_20230131_0554_7e5a5aa.npz' # ZIF8 MT EXP
# fileParameter = 'zlcParameters_20210824_1522_c8173b1.npz'
# fileParameter = 'zlcParameters_20230216_2106_7e5a5aa.npz' # ZIF8 MCB30 EXP
# fileParameter = 'zlcParameters_20230216_0638_7e5a5aa.npz' # ZIF8 MCB20 EXP
# fileParameter = 'zlcParameters_20230221_0245_7e5a5aa.npz' # ZIF8 MT EXP LOW FLOW
# fileParameter = 'zlcParameters_20230222_0332_7e5a5aa.npz' # ZIF8 MT EXP LOW FLOW
# fileParameter = 'zlcParameters_20230228_0507_7e5a5aa.npz' # ZIF8 MCB20 EXP LOW FLOW
# fileParameter = 'zlcParameters_20230225_1202_7e5a5aa.npz' # ZIF8 MCB30 EXP LOW FLOW
# fileParameter = 'zlcParameters_20230312_2111_7e5a5aa.npz' # ZYH EXP ALL FLOW SSL HIGH COMP
fileParameter = 'zlcParameters_20230314_0031_7e5a5aa.npz' # ZYH EXP ALL FLOW DSL HIGH COMP
# fileParameter = 'zlcParameters_20230315_0130_59cc206.npz' # ZYH EXP ALL FLOW SSS ALL COMP
# fileParameter = 'zlcParameters_20230315_0921_59cc206.npz' # ZYH EXP ALL FLOW SSS ALL COMP
fileParameter = 'zlcParameters_20230318_1112_59cc206.npz' # ZZYH ALL FLOW KIN ONLY
# fileParameter = 'zlcParameters_20230324_0100_59cc206.npz' # ZZYH ALL FLOW KIN ONLY HIGH COMP
fileParameter = 'zlcParameters_20230330_1528_59cc206.npz' # ZZYNa ALL FLOW KIN ONLY HIGH COMP
# deadVolumeFile = [[['deadVolumeCharacteristics_20230321_1048_59cc206.npz', # 50A
#                     'deadVolumeCharacteristics_20230321_1238_59cc206.npz']], # 51A
#                   [['deadVolumeCharacteristics_20230321_1137_59cc206.npz', # 50B
#                     'deadVolumeCharacteristics_20230321_1252_59cc206.npz']]] # 51B
# fileParameter = 'deadVolumeCharacteristics_20220726_0235_e81a19e.npz'
# fileParameter = 'deadVolumeCharacteristics_20220714_2133_6072a85.npz'
# fileParameter = 'deadVolumeCharacteristics_20220712_1444_e81a19e.npz'
# fileParameter = 'deadVolumeCharacteristics_20230220_1813_7e5a5aa.npz'
# fileParameter = 'deadVolumeCharacteristics_20230220_1752_7e5a5aa.npz'
# fileParameter = 'deadVolumeCharacteristics_20230220_1954_7e5a5aa.npz'
# fileParameter = 'deadVolumeCharacteristics_20230309_1626_7e5a5aa.npz' 
# fileParameter = 'deadVolumeCharacteristics_20230821_1834_b571c46.npz'
# fileParameter = 'deadVolumeCharacteristicY:/ha3215/home/ERASE/simulationResults/deadVolumeCharacteristics_20230821_1639_b571c46.npzs_20230321_1137_59cc206.npz'
    # deadVolumeFile = [[['deadVolumeCharacteristics_20230821_1803_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_1849_b571c46.npz']],
    #                   [['deadVolumeCharacteristics_20230821_1813_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_1909_b571c46.npz']]] #highflow CMS Ar
    
    
fileParameter = 'zlcParameters_ZYH_20230908_1249_b571c46.npz' # ZYH ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_ZYNa_20230909_0021_b571c46.npz' # ZYNa ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_ZYTMA_20230910_0442_b571c46.npz' # ZYTMA ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_CMS3KAr_20230911_0436_b571c46.npz' # CMS Ar ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_CMS3K_20230913_0550_b571c46.npz' # CMS He ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_Zeolite13X_20230913_0352_b571c46.npz' # 13X ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_CMS3KAr_20230917_0304_b571c46.npz' # CMS Ar ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_CMS3K_20230916_0105_b571c46.npz' # CMS He ALL FLOW SBMACRO

fileParameter = 'zlcParameters_ZYH_20230914_0359_b571c46.npz' # ZYH ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_ZYNa_20230914_1950_b571c46.npz' # ZYNa ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_ZYTMA_20230915_1651_b571c46.npz' # ZYTMA ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_Zeolite13X_20230918_0651_b571c46.npz' # 13X ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_ActivatedCarbon_20230918_1543_b571c46.npz' # AC ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_CMS3K_20230919_1800_b571c46.npz' # CMS He ALL FLOW SBMACRO high comp
# fileParameter = 'zlcParameters_CMS3KAr_20230920_0458_b571c46.npz' # CMS Ar ALL FLOW SBMACRO high comp
# fileParameter = 'zlcParameters_ActivatedCarbon_20230921_0603_b571c46.npz' # AC ALL FLOW SBMACRO high comp


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
    # rawFileName = ['ZLC_DeadVolume_Exp26A_Output.mat',
    #                'ZLC_DeadVolume_Exp26B_Output.mat',
    #                'ZLC_DeadVolume_Exp26C_Output.mat',
    #                'ZLC_DeadVolume_Exp26D_Output.mat',]
    rawFileName = ['ZLC_Empty_Exp14A_Output.mat',
                'ZLC_Empty_Exp15A_Output.mat',
                'ZLC_Empty_Exp16A_Output.mat',
                'ZLC_Empty_Exp17A_Output.mat',
	            'ZLC_Empty_Exp14B_Output.mat',
                'ZLC_Empty_Exp15B_Output.mat',
                'ZLC_Empty_Exp16B_Output.mat',
                'ZLC_Empty_Exp17B_Output.mat',]
    
    rawFileName = ['ZLC_Empty_Exp10A_Output.mat',
                'ZLC_Empty_Exp11A_Output.mat',
                'ZLC_Empty_Exp12A_Output.mat',
                'ZLC_Empty_Exp13A_Output.mat',
 	            'ZLC_Empty_Exp10B_Output.mat',
                'ZLC_Empty_Exp11B_Output.mat',
                'ZLC_Empty_Exp12B_Output.mat',
                'ZLC_Empty_Exp13B_Output.mat',]
    
    # rawFileName = ['ZLC_Empty_Exp13A_Output.mat',
    #             'ZLC_Empty_Exp13B_Output.mat']
    # rawFileName = ['ZLC_Empty_Exp10A_Output.mat',
    #                'ZLC_Empty_Exp10B_Output.mat']
    
    # rawFileName = ['ZLC_Empty_Exp17A_Output.mat',
    #             'ZLC_Empty_Exp17B_Output.mat']
    # rawFileName = ['ZLC_Empty_Exp14A_Output.mat',
    #                 'ZLC_Empty_Exp14B_Output.mat']
    # rawFileName =['ZLC_Empty_Exp50A_Output.mat',
    #             'ZLC_Empty_Exp50B_Output.mat']
    # rawFileName =['ZLC_Empty_Exp51A_Output.mat',
    #             'ZLC_Empty_Exp51B_Output.mat']
    rawFileName =['ZLC_Empty_Exp50A_Output.mat']
    
    # rawFileName =['ZLC_Empty_Exp50B_Output.mat',
    #             'ZLC_Empty_Exp58B_Output.mat']
    # rawFileName = ['ZLC_EmptyAr_Exp01A_Output.mat',
    #             'ZLC_EmptyAr_Exp01B_Output.mat',
    #             'ZLC_EmptyAr_Exp02A_Output.mat',
    #             'ZLC_EmptyAr_Exp02B_Output.mat',]
    rawFileName = ['ZLC_EmptyAr_Exp01B_Output.mat',]
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
    # x =  [1.95749126e+00 ,1.51763159e+00 ,9.16424097e-01, 19, 1.26025728e-02]
    # x = [2.16121072, 0.97641732 ,1.12077131 ,6.  ,       0.01729448]
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
    # downsampleInt = np.array([2])

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
                          color=colorsForPlot[ii],alpha=0.05,label=str(round(np.mean(flowRateExp[-1:-10:-1]),2))+" ccs") # Experimental response
            if simulateModel:
                ax1.plot(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii]) # Simulation response    
            ax1.set(xlabel='$t$ [s]', 
                    ylabel='$y_1$ [-]',
                    xlim = [0,200], ylim = [0, 1])   
            ax1.locator_params(axis="x", nbins=5)
            ax1.locator_params(axis="y", nbins=5)
            ax1.legend()
    
            # Log scale
            ax2.scatter(timeElapsedExp,moleFracExp,
                          marker = markerForPlot[ii],
                          color='none', edgecolors = colorsForPlot[ii],alpha=0.05,label=str(round(flowRateExp[-1],2))+" ccs", s =20, linewidth = 1, linestyle = '-') # Experimental response
            if simulateModel:
                ax2.semilogy(timeElapsedExp,moleFracSim,
                              color=colorsForPlot[ii], linewidth = 1) # Simulation response
            ax2.set(xlabel='$t$ [s]', 
                    xlim = [0,200], ylim = [1e-2, 1])   
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
                          color=colorsForPlot[ii],alpha=0.01,label=str(round(np.mean(flowRateDV),2))+" ccs") # Experimental response
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
    # rawFileName = ['ZLC_ActivatedCarbon_Exp72A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp74A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp76A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp72B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp74B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp76B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp73A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp75A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp77A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp73B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp75B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Exp77B_Output.mat',]
    
    # rawFileName = ['ZLC_ActivatedCarbon_Sim01A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim03A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim05A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim01B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim03B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim05B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim02A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim04A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim06A_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim02B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim04B_Output.mat',
    #                 'ZLC_ActivatedCarbon_Sim06B_Output.mat',]

    # rawFileName =  ['ZLC_BoronNitride_Exp34A_Output.mat',
    #             'ZLC_BoronNitride_Exp36A_Output.mat',
    #             'ZLC_BoronNitride_Exp38A_Output.mat',
    #             'ZLC_BoronNitride_Exp34B_Output.mat',
    #             'ZLC_BoronNitride_Exp36B_Output.mat',
    #             'ZLC_BoronNitride_Exp38B_Output.mat',
    #             'ZLC_BoronNitride_Exp35A_Output.mat',
    #             'ZLC_BoronNitride_Exp37A_Output.mat',
    #             'ZLC_BoronNitride_Exp39A_Output.mat',
    #             'ZLC_BoronNitride_Exp35B_Output.mat',
    #             'ZLC_BoronNitride_Exp37B_Output.mat',
    #             'ZLC_BoronNitride_Exp39B_Output.mat',]
    
    # rawFileName = ['ZLC_BoronNitride_Sim01A_Output.mat',
    #                 'ZLC_BoronNitride_Sim03A_Output.mat',
    #                 'ZLC_BoronNitride_Sim05A_Output.mat',
    #                 'ZLC_BoronNitride_Sim01B_Output.mat',
    #                 'ZLC_BoronNitride_Sim03B_Output.mat',
    #                 'ZLC_BoronNitride_Sim05B_Output.mat',
    #                 'ZLC_BoronNitride_Sim02A_Output.mat',
    #                 'ZLC_BoronNitride_Sim04A_Output.mat',
    #                 'ZLC_BoronNitride_Sim06A_Output.mat',
    #                 'ZLC_BoronNitride_Sim02B_Output.mat',
    #                 'ZLC_BoronNitride_Sim04B_Output.mat',
    #                 'ZLC_BoronNitride_Sim06B_Output.mat',]

    # rawFileName = ['ZLC_Zeolite13X_Sim01A_Output.mat',
    #                 'ZLC_Zeolite13X_Sim03A_Output.mat',
    #                 'ZLC_Zeolite13X_Sim05A_Output.mat',
    #                 'ZLC_Zeolite13X_Sim01B_Output.mat',
    #                 'ZLC_Zeolite13X_Sim03B_Output.mat',
    #                 'ZLC_Zeolite13X_Sim05B_Output.mat',
    #                 'ZLC_Zeolite13X_Sim02A_Output.mat',
    #                 'ZLC_Zeolite13X_Sim04A_Output.mat',
    #                 'ZLC_Zeolite13X_Sim06A_Output.mat',
    #                 'ZLC_Zeolite13X_Sim02B_Output.mat',
    #                 'ZLC_Zeolite13X_Sim04B_Output.mat',
    #                 'ZLC_Zeolite13X_Sim06B_Output.mat',]

    # rawFileName = ['ZLC_ZIF8_MT_Exp01A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp03A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp07A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp01B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp03B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp07B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp02A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp04A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp08A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp02B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp04B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp08B_Output.mat',]    
    # rawFileName = ['ZLC_ZIF8_MT_Exp11A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp05A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp09A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp11B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp05B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp09B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp12A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp06A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp10A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp12B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp06B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp10B_Output.mat',]
    # rawFileName = ['ZLC_ZIF8_MT_Exp01A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp05A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp09A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp01B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp05B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp09B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp02A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp06A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp10A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp02B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp06B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp10B_Output.mat',]
    # rawFileName = ['ZLC_ZIF8_MT_Exp01A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp03A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp07A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp01B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp03B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp07B_Output.mat',]
    # rawFileName = ['ZLC_ZIF8_MT_Exp02A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp04A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp08A_Output.mat',
    #         'ZLC_ZIF8_MT_Exp02B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp04B_Output.mat',
    #         'ZLC_ZIF8_MT_Exp08B_Output.mat',]

    # rawFileName = ['ZLC_ZIF8_MCB30_Exp01A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp03A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp07A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp01B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp03B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp07B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp02A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp04A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp08A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp02B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp04B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp08B_Output.mat',]
    
    # rawFileName = ['ZLC_ZIF8_MCB30_Exp01A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp03A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp07A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp01B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp03B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp07B_Output.mat',]
    
    # rawFileName = ['ZLC_ZIF8_MCB30_Exp02A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp04A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp08A_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp02B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp04B_Output.mat',
    #         'ZLC_ZIF8_MCB30_Exp08B_Output.mat',]
    
    rawFileName = ['ZLC_ZIF8_MCB20_Exp01A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp05A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp07A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp01B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp05B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp07B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp02A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp06A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp08A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp02B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp06B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp08B_Output.mat',]
    
    rawFileName = ['ZLC_ZYH_Exp09A_Output.mat',
            'ZLC_ZYH_Exp11A_Output.mat',
            'ZLC_ZYH_Exp13A_Output.mat',
            'ZLC_ZYH_Exp09B_Output.mat',
            'ZLC_ZYH_Exp11B_Output.mat',
            'ZLC_ZYH_Exp13B_Output.mat',
            'ZLC_ZYH_Exp10A_Output.mat',
            'ZLC_ZYH_Exp12A_Output.mat',
            'ZLC_ZYH_Exp14A_Output.mat',
            'ZLC_ZYH_Exp10B_Output.mat',
            'ZLC_ZYH_Exp12B_Output.mat',
            'ZLC_ZYH_Exp14B_Output.mat',]
    
    # rawFileName = ['ZLC_ZYH_Exp09A_Output.mat',
    #         'ZLC_ZYH_Exp11A_Output.mat',
    #         'ZLC_ZYH_Exp13A_Output.mat',
    #         'ZLC_ZYH_Exp09B_Output.mat',
    #         'ZLC_ZYH_Exp11B_Output.mat',
    #         'ZLC_ZYH_Exp13B_Output.mat',]
    
    # rawFileName  = ['ZLC_ZYH_Exp05B_Output.mat',
    #         'ZLC_ZYH_Exp03B_Output.mat',
    #         'ZLC_ZYH_Exp01B_Output.mat',
    #         'ZLC_ZYH_Exp06B_Output.mat',
    #         'ZLC_ZYH_Exp04B_Output.mat',
    #         'ZLC_ZYH_Exp02B_Output.mat',]
    
    # rawFileName = ['ZLC_ZYH_Exp10A_Output.mat',
    #         'ZLC_ZYH_Exp12A_Output.mat',
    #         'ZLC_ZYH_Exp14A_Output.mat',
    #         'ZLC_ZYH_Exp10B_Output.mat',
    #         'ZLC_ZYH_Exp12B_Output.mat',
    #         'ZLC_ZYH_Exp14B_Output.mat',]
    
    # rawFileName  = ['ZLC_ZYH_Exp09B_Output.mat',
    #         'ZLC_ZYH_Exp11B_Output.mat',
    #         'ZLC_ZYH_Exp13B_Output.mat',
    #         'ZLC_ZYH_Exp10B_Output.mat',
    #         'ZLC_ZYH_Exp12B_Output.mat',
    #         'ZLC_ZYH_Exp14B_Output.mat',]
    # rawFileName  = ['ZLC_ZYH_Exp09A_Output.mat',
    #         'ZLC_ZYH_Exp11A_Output.mat',
    #         'ZLC_ZYH_Exp13A_Output.mat',
    #         'ZLC_ZYH_Exp10A_Output.mat',
    #         'ZLC_ZYH_Exp12A_Output.mat',
    #         'ZLC_ZYH_Exp14A_Output.mat',]
    # rawFileName  = ['ZLC_ZYH_Exp09B_Output.mat',
    #         'ZLC_ZYH_Exp11B_Output.mat',
    #         'ZLC_ZYH_Exp13B_Output.mat',]
    # rawFileName  = ['ZLC_ZYH_Exp10B_Output.mat',
    #         'ZLC_ZYH_Exp12B_Output.mat',
    #         'ZLC_ZYH_Exp14B_Output.mat',]
    # rawFileName  = ['ZLC_ZYH_Exp10B_Output.mat',
    #         'ZLC_ZYH_Exp12B_Output.mat',
    #         'ZLC_ZYH_Exp14B_Output.mat',]
    
    # rawFileName  = ['ZLC_ZYH_Exp09B_Output.mat',
    #         'ZLC_ZYH_Exp11B_Output.mat',
    #         'ZLC_ZYH_Exp13B_Output.mat',]
    
    # rawFileName = ['ZLC_ZYH_Exp09A_Output.mat',
    #         'ZLC_ZYH_Exp11A_Output.mat',
    #         'ZLC_ZYH_Exp13A_Output.mat',
    #         'ZLC_ZYH_Exp09B_Output.mat',
    #         'ZLC_ZYH_Exp11B_Output.mat',
    #         'ZLC_ZYH_Exp13B_Output.mat',]
      
    rawFileName = ['ZLC_ZYNa_Exp05A_Output.mat',
            'ZLC_ZYNa_Exp03A_Output.mat',
            'ZLC_ZYNa_Exp09A_Output.mat',
            'ZLC_ZYNa_Exp05B_Output.mat',
            'ZLC_ZYNa_Exp03B_Output.mat',
            'ZLC_ZYNa_Exp09B_Output.mat',
            'ZLC_ZYNa_Exp06A_Output.mat',
            'ZLC_ZYNa_Exp04A_Output.mat',
            'ZLC_ZYNa_Exp10A_Output.mat',
            'ZLC_ZYNa_Exp06B_Output.mat',
            'ZLC_ZYNa_Exp04B_Output.mat',
            'ZLC_ZYNa_Exp10B_Output.mat',]  
    
    # rawFileName = ['ZLC_ZYNa_Exp05B_Output.mat',
    #         'ZLC_ZYNa_Exp03B_Output.mat',
    #         'ZLC_ZYNa_Exp09B_Output.mat',
    #         'ZLC_ZYNa_Exp06B_Output.mat',
    #         'ZLC_ZYNa_Exp04B_Output.mat',
    #         'ZLC_ZYNa_Exp10B_Output.mat',] 
    # # rawFileName = ['ZLC_ZYNa_Exp05B_Output.mat',
    #         'ZLC_ZYNa_Exp03B_Output.mat',
    #         'ZLC_ZYNa_Exp09B_Output.mat',]
    # rawFileName = ['ZLC_ZYNa_Exp06B_Output.mat',
    #         'ZLC_ZYNa_Exp04B_Output.mat',
    #         'ZLC_ZYNa_Exp10B_Output.mat',]
    # rawFileName = ['ZLC_ZYNa_Exp03B_Output.mat',
    #         'ZLC_ZYNa_Exp04B_Output.mat',] 
    # rawFileName = ['ZLC_ZYNa_Exp05B_Output.mat',
    #         'ZLC_ZYNa_Exp03B_Output.mat',
    #         'ZLC_ZYNa_Exp09B_Output.mat',]
    # rawFileName = ['ZLC_ZYNa_Exp05A_Output.mat',
    #         'ZLC_ZYNa_Exp03A_Output.mat',
    #         'ZLC_ZYNa_Exp09A_Output.mat',
    #         'ZLC_ZYNa_Exp06A_Output.mat',
    #         'ZLC_ZYNa_Exp04A_Output.mat',
    #         'ZLC_ZYNa_Exp10A_Output.mat',]
    # rawFileName = ['ZLC_ZYNa_Exp05A_Output.mat',
    #         'ZLC_ZYNa_Exp03A_Output.mat',
    #         'ZLC_ZYNa_Exp09A_Output.mat',
    #         'ZLC_ZYNa_Exp05B_Output.mat',
    #         'ZLC_ZYNa_Exp03B_Output.mat',
    #         'ZLC_ZYNa_Exp09B_Output.mat',]
    
    # rawFileName = ['ZLC_ZYNa_Exp06A_Output.mat',
    #         'ZLC_ZYNa_Exp04A_Output.mat',
    #         'ZLC_ZYNa_Exp02A_Output.mat',
    #         'ZLC_ZYNa_Exp06B_Output.mat',
    #         'ZLC_ZYNa_Exp04B_Output.mat',
    #         'ZLC_ZYNa_Exp02B_Output.mat',]
    
    # rawFileName = ['ZLC_ZYTMA_Exp07B_Output.mat',
    #         'ZLC_ZYTMA_Exp09B_Output.mat',
    #         'ZLC_ZYTMA_Exp11B_Output.mat',
    #         'ZLC_ZYTMA_Exp08B_Output.mat',
    #         'ZLC_ZYTMA_Exp10B_Output.mat',
    #         'ZLC_ZYTMA_Exp12B_Output.mat',] 
    
    rawFileName = ['ZLC_ZYTMA_Exp01A_Output.mat',
            'ZLC_ZYTMA_Exp03A_Output.mat',
            'ZLC_ZYTMA_Exp05A_Output.mat',
            'ZLC_ZYTMA_Exp01B_Output.mat',
            'ZLC_ZYTMA_Exp03B_Output.mat',
            'ZLC_ZYTMA_Exp05B_Output.mat',
            'ZLC_ZYTMA_Exp02A_Output.mat',
            'ZLC_ZYTMA_Exp04A_Output.mat',
            'ZLC_ZYTMA_Exp06A_Output.mat',
            'ZLC_ZYTMA_Exp02B_Output.mat',
            'ZLC_ZYTMA_Exp04B_Output.mat',
            'ZLC_ZYTMA_Exp06B_Output.mat',] 
    
    # rawFileName = ['ZLC_ZYTMA_Exp01B_Output.mat',
    #         'ZLC_ZYTMA_Exp03B_Output.mat',
    #         'ZLC_ZYTMA_Exp05B_Output.mat',
    #         'ZLC_ZYTMA_Exp02B_Output.mat',
    #         'ZLC_ZYTMA_Exp04B_Output.mat',
    #         'ZLC_ZYTMA_Exp06B_Output.mat',]     
    # rawFileName = ['ZLC_ZYTMA_Exp01B_Output.mat',
    #         'ZLC_ZYTMA_Exp03B_Output.mat',
    #         'ZLC_ZYTMA_Exp05B_Output.mat',]     
    # rawFileName = ['ZLC_ZYTMA_Exp02B_Output.mat',
    #         'ZLC_ZYTMA_Exp04B_Output.mat',
    #         'ZLC_ZYTMA_Exp06B_Output.mat',] 
    # rawFileName = ['ZLC_ZYTMA_Exp01A_Output.mat',
    #         'ZLC_ZYTMA_Exp03A_Output.mat',
    #         'ZLC_ZYTMA_Exp05A_Output.mat',
    #         'ZLC_ZYTMA_Exp02A_Output.mat',
    #         'ZLC_ZYTMA_Exp04A_Output.mat',
    #         'ZLC_ZYTMA_Exp06A_Output.mat',] 
    # rawFileName = ['ZLC_ZYTMA_Exp05B_Output.mat',
    #                'ZLC_ZYTMA_Exp06B_Output.mat',]
    
    # rawFileName = ['ZLC_ZIF8_MCB20_Exp01A_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp05A_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp07A_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp01B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp05B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp07B_Output.mat',]
    # rawFileName = ['ZLC_ZIF8_MCB20_Exp02A_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp06A_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp08A_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp02B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp06B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp08B_Output.mat',]
    # rawFileName = ['ZLC_ZIF8_MCB20_Exp01B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp05B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp07B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp02B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp06B_Output.mat',
    #         'ZLC_ZIF8_MCB20_Exp08B_Output.mat',]
    # rawFileName =  ['ZLC_BNpFAS_Exp01A_Output.mat',
 	  #           'ZLC_BNpFAS_Exp03A_Output.mat',
 	  #           'ZLC_BNpFAS_Exp05A_Output.mat',
 	  #           'ZLC_BNpFAS_Exp01B_Output.mat',
 	  #           'ZLC_BNpFAS_Exp03B_Output.mat',
 	  #           'ZLC_BNpFAS_Exp05B_Output.mat',
 	  #           'ZLC_BNpFAS_Exp02A_Output.mat',
 	  #           'ZLC_BNpFAS_Exp04A_Output.mat',
 	  #           'ZLC_BNpFAS_Exp06A_Output.mat',
 	  #           'ZLC_BNpFAS_Exp02B_Output.mat',
 	  #           'ZLC_BNpFAS_Exp04B_Output.mat',
    #         'ZLC_BNpFAS_Exp06B_Output.mat',]
    
#     rawFileName =         ['ZLC_BNFASp_Exp35A_Output.mat',
# 		            'ZLC_BNFASp_Exp37A_Output.mat',
# 		            'ZLC_BNFASp_Exp33A_Output.mat',
# 		            'ZLC_BNFASp_Exp36A_Output.mat',
# 		            'ZLC_BNFASp_Exp38A_Output.mat',
# 		            'ZLC_BNFASp_Exp34A_Output.mat',
#                     'ZLC_BNFASp_Exp35B_Output.mat',
# 		            'ZLC_BNFASp_Exp37B_Output.mat',
# 		            'ZLC_BNFASp_Exp33B_Output.mat',
# 		            'ZLC_BNFASp_Exp36B_Output.mat',
# 		            'ZLC_BNFASp_Exp38B_Output.mat',
# 		            'ZLC_BNFASp_Exp34B_Output.mat',]
    
#     rawFileName =         ['ZLC_BNFASp_Exp35A_Output.mat',
# 		            'ZLC_BNFASp_Exp27A_Output.mat',
# 		            'ZLC_BNFASp_Exp33A_Output.mat',
#                     'ZLC_BNFASp_Exp35B_Output.mat',
# 		            'ZLC_BNFASp_Exp27B_Output.mat',
# 		            'ZLC_BNFASp_Exp33B_Output.mat',]

#     rawFileName = ['ZLC_Lewatit_DA_Exp05A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp07A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp09A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp11A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp13A_Output.mat',
#                     'ZLC_Lewatit_DA_Exp05B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp07B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp09B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp11B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp13B_Output.mat',]

    # rawFileName  = ['ZLC_CMS3K_Exp05_Output.mat',
    #         'ZLC_CMS3K_Exp03B_Output.mat',
    #         'ZLC_CMS3K_Exp01B_Output.mat',
    #         'ZLC_CMS3K_Exp05B_Output.mat',
    #         'ZLC_CMS3K_Exp03B_Output.mat',
    #         'ZLC_CMS3K_Exp01B_Output.mat',]
    
    # rawFileName  = ['ZLC_CMS3K_Exp11B_Output.mat',
    #         'ZLC_CMS3K_Exp09B_Output.mat',
    #         'ZLC_CMS3K_Exp07B_Output.mat',]
    
    # rawFileName  = ['ZLC_CMS3K_Exp13A_Output.mat',
    #         'ZLC_CMS3K_Exp17A_Output.mat',
    #         'ZLC_CMS3K_Exp15A_Output.mat',
    #         'ZLC_CMS3K_Exp13B_Output.mat',
    #         'ZLC_CMS3K_Exp17B_Output.mat',
    #         'ZLC_CMS3K_Exp15B_Output.mat',]
    
    # rawFileName = ['ZLC_CMS3KAr_Exp01A_Output.mat',
    #         'ZLC_CMS3KAr_Exp03A_Output.mat',
    #         'ZLC_CMS3KAr_Exp05A_Output.mat',
    #         'ZLC_CMS3KAr_Exp02A_Output.mat',
    #         'ZLC_CMS3KAr_Exp04A_Output.mat',
    #         'ZLC_CMS3KAr_Exp06A_Output.mat',
    #         'ZLC_CMS3KAr_Exp01B_Output.mat',
    #         'ZLC_CMS3KAr_Exp03B_Output.mat',
    #         'ZLC_CMS3KAr_Exp05B_Output.mat',
    #         'ZLC_CMS3KAr_Exp02B_Output.mat',
    #         'ZLC_CMS3KAr_Exp04B_Output.mat',
    #         'ZLC_CMS3KAr_Exp06B_Output.mat',]
    
    # rawFileName  = ['ZLC_CMS3K_Exp13A_Output.mat',
    #         'ZLC_CMS3K_Exp17A_Output.mat',
    #         'ZLC_CMS3K_Exp15A_Output.mat',
    #         'ZLC_CMS3K_Exp14A_Output.mat',
    #         'ZLC_CMS3K_Exp18A_Output.mat',
    #         'ZLC_CMS3K_Exp16A_Output.mat',
    #         'ZLC_CMS3K_Exp13B_Output.mat',
    #         'ZLC_CMS3K_Exp17B_Output.mat',
    #         'ZLC_CMS3K_Exp15B_Output.mat',
    #         'ZLC_CMS3K_Exp14B_Output.mat',
    #         'ZLC_CMS3K_Exp18B_Output.mat',
    #         'ZLC_CMS3K_Exp16B_Output.mat',]
    
    # rawFileName = ['ZLC_CMS3KAr_Exp01A_Output.mat',
    #         'ZLC_CMS3KAr_Exp03A_Output.mat',
    #         'ZLC_CMS3KAr_Exp05A_Output.mat',
    #         'ZLC_CMS3KAr_Exp01B_Output.mat',
    #         'ZLC_CMS3KAr_Exp03B_Output.mat',
    #         'ZLC_CMS3KAr_Exp05B_Output.mat',]
    
    # rawFileName = ['ZLC_CMS3KAr_Exp02A_Output.mat',
    #         'ZLC_CMS3KAr_Exp04A_Output.mat',
    #         'ZLC_CMS3KAr_Exp06A_Output.mat',
    #         'ZLC_CMS3KAr_Exp02B_Output.mat',
    #         'ZLC_CMS3KAr_Exp04B_Output.mat',
    #         'ZLC_CMS3KAr_Exp06B_Output.mat',]
    
    
    # rawFileName  = ['ZLC_CMS3K_Exp13A_Output.mat',
    #         'ZLC_CMS3K_Exp17A_Output.mat',
    #         'ZLC_CMS3K_Exp15A_Output.mat',
    #         'ZLC_CMS3K_Exp13B_Output.mat',
    #         'ZLC_CMS3K_Exp17B_Output.mat',
    #         'ZLC_CMS3K_Exp15B_Output.mat',]
    
    # rawFileName  = ['ZLC_CMS3K_Exp14A_Output.mat',
    #         'ZLC_CMS3K_Exp18A_Output.mat',
    #         'ZLC_CMS3K_Exp16A_Output.mat',
    #         'ZLC_CMS3K_Exp14B_Output.mat',
    #         'ZLC_CMS3K_Exp18B_Output.mat',
    #         'ZLC_CMS3K_Exp16B_Output.mat',]


    ################## DV MASS BALANCE ########################
    rawFileNameDV = ['ZLC_Empty_Exp10A_Output.mat',
                     'ZLC_Empty_Exp10A_Output.mat',
                     'ZLC_Empty_Exp10A_Output.mat',
                     'ZLC_Empty_Exp10B_Output.mat',
                     'ZLC_Empty_Exp10B_Output.mat',
                     'ZLC_Empty_Exp10B_Output.mat',
                     'ZLC_Empty_Exp13A_Output.mat',
                     'ZLC_Empty_Exp13A_Output.mat',
                     'ZLC_Empty_Exp13A_Output.mat',
                     'ZLC_Empty_Exp13B_Output.mat',
                     'ZLC_Empty_Exp13B_Output.mat',
                     'ZLC_Empty_Exp13B_Output.mat',]
    rawFileNameDV = ['ZLC_Empty_Exp14A_Output.mat',
                     'ZLC_Empty_Exp14A_Output.mat',
                     'ZLC_Empty_Exp14A_Output.mat',
                     'ZLC_Empty_Exp14B_Output.mat',
                     'ZLC_Empty_Exp14B_Output.mat',
                     'ZLC_Empty_Exp14B_Output.mat',
                     'ZLC_Empty_Exp17A_Output.mat',
                     'ZLC_Empty_Exp17A_Output.mat',
                     'ZLC_Empty_Exp17A_Output.mat',
                     'ZLC_Empty_Exp17B_Output.mat',
                     'ZLC_Empty_Exp17B_Output.mat',
                     'ZLC_Empty_Exp17B_Output.mat',]
    
    rawFileNameDV = ['ZLC_Empty_Exp50A_Output.mat',
                     'ZLC_Empty_Exp50A_Output.mat',
                     'ZLC_Empty_Exp50A_Output.mat',
                     'ZLC_Empty_Exp50B_Output.mat',
                     'ZLC_Empty_Exp50B_Output.mat',
                     'ZLC_Empty_Exp50B_Output.mat',
                     'ZLC_Empty_Exp51A_Output.mat',
                     'ZLC_Empty_Exp51A_Output.mat',
                     'ZLC_Empty_Exp51A_Output.mat',
                     'ZLC_Empty_Exp51B_Output.mat',
                     'ZLC_Empty_Exp51B_Output.mat',
                     'ZLC_Empty_Exp51B_Output.mat',]
    
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,os.path.join('..','experimental','runData'),rawFileNameDV,'DV')
    # Get the processed file names
    fileNameDV = filesToProcess(False,[],[],'DV')
    
    numPointsExp = np.zeros(len(fileNameDV))
    deadVolumeIntegral = np.zeros(len(fileNameDV))
    for ii in range(len(fileNameDV)): 
        fileToLoad = fileNameDV[ii]
        # Load experimental molefraction
        timeElapsedExpDV = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExpDV)
    
    # Downsample intervals
    downsampleInt = numPointsExp/numPointsExp
    # Loop over all available files    
    for ii in range(len(fileNameDV)):
        # Initialize outputs
        moleFracSim = []
        # Path of the file name
        fileToLoad = fileNameDV[ii]   
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
        flowRateTemp = load(fileToLoad)["flowRate"].flatten()
        timeElapsedExpDV = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
        moleFracExpDV = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
        flowRateExpDV = flowRateTemp[::int(np.round(downsampleInt[ii]))]
        # flowInDV = np.zeros((len(flowRateExpDV)))          
        flowInDV = flowRateExpDV       
        # flowInDV[:] = np.mean(flowRateExpDV[-1:-2:-1])
        deadVolumeIntegral[ii] = np.trapz(moleFracExpDV,np.multiply(flowInDV*1e-6,timeElapsedExpDV))
        
        # Integration and ode evaluation time
        timeInt = timeElapsedExpDV
    ###########################################################
        
    # ZLC parameter model path
    parameterPath = os.path.join('..','simulationResults',fileParameter)
    
    # Temperature (for each experiment)
    # temperatureExp = [344.69, 325.39, 306.15]*4 # AC Experiments
    # temperatureExp = [308.15, 328.15, 348.15]*4 # AC Simulations
    # temperatureExp = [344.6, 325.49, 306.17,]*4 # BN (2 pellets) Experiments
    # temperatureExp = [308.15, 328.15, 348.15]*4 # BN (2 pellets) Simulations 
    # temperatureExp = [ 303.15, 293.15, 283.15, ]*4 # ZIF8 
    # temperatureExp = [ 283.15, 293.15, 303.15,]*4 # BNFAS
    # temperatureExp = [ 363.15, 348.15, 333.15, 318.15, 303.15,]*2 # lewatit
    # temperatureExp = [ 288.15, 298.15, 308.15, ]*4 # ZY
    # temperatureExp = [ 288.15 ]*4 # ZY 
    # temperatureExp = [312.15]*4 # ZY    temperatureExp = [ 288.15 ]*4 # ZY 

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
    
    ## Adsorbent properties
    # Adsorbent density [kg/m3]
    # This has to be the skeletal density
    # adsorbentDensity = 1680 # Activated carbon skeletal density [kg/m3]
    # adsorbentDensity = 4100 # Zeolite 13X H 
    # adsorbentDensity = 1250 # BNFASp skeletal density [kg/m3]
    # adsorbentDensity = 2320 # BNpFAS skeletal density [kg/m3]
    # adsorbentDensity = 1060 # lewatit skeletal density [kg/m3]
    # adsorbentDensity = 988 # Lewatit skeletal density [kg/m3]
    # adsorbentDensity = 1555 # ZIF-8 MT
    # adsorbentDensity = 2400 # ZIF-8 MCB20
    # adsorbentDensity = 2100 # ZIF-8 MCB30
    # adsorbentDensity = 1020 # ZYH
    # adsorbentDensity = 3200 # ZYNa
    # adsorbentDensity = 2890 # ZYTMA
    # adsorbentDensity = 2130 # ZYH MSB
    # adsorbentDensity = 2410 # ZYNa MSB
    # adsorbentDensity = 2310 # ZYTMA MSB
    # adsorbentDensity = 1680 # CMS 3K

    # Particle porosity
    # particleEpsilon = 0.61 # AC
    # particleEpsilon = 0.79 # Zeolite 13X H
    # particleEpsilon = 0.64 # BNFASp
    # particleEpsilon = 0.67 # BNpFAS
    # particleEpsilon = 0.337 # lewatit
    # particleEpsilon = 0.44 # Lewatit
    # particleEpsilon = 0.67 # BNpFAS
    # particleEpsilon = 0.47 # ZIF-8 MT
    # particleEpsilon = 0.62 # ZIF-8 MCB20
    # particleEpsilon = 0.59 # ZIF-8 MCB30
    # particleEpsilon = 0.90 # ZYH
    # particleEpsilon = 0.76 # ZYNa
    # particleEpsilon = 0.71 # ZYTMA
    # particleEpsilon = 0.61 # CMS 3K

    # Particle mass [g]
    # massSorbent = 0.0625  # AC
    # massSorbent = 0.0594 # Zeolite 13X H
    # massSorbent = 0.069  # BNFASp
    # massSorbent = 0.1  # BNFASp
    # massSorbent = 0.0262  # lewatit
    # massSorbent = 0.0262  # Lewatit
    # massSorbent = 0.1295  # BNpFAS
    # massSorbent = 0.059 # ZIF-8 MT
    # massSorbent = 0.075 # ZIF-8 MCB20
    # massSorbent = 0.102 # ZIF-8 MCB30
    # massSorbent = 0.086 # ZYH real
    # massSorbent = 0.085 # ZYH (v1)
    # massSorbent = 0.07 # ZYNa
    # massSorbent = 0.065 # ZYTMA
    # massSorbent = 0.077 # CMS 3K
    # massSorbent = 0.072 # CMS 3K
    # massSorbent = 0.061 # CMS 3K


    # Volume of sorbent material [m3]
    volSorbent = (massSorbent/1000)/adsorbentDensity
    
    # Volume of gas chamber (dead volume) [m3]
    volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

    # Dead volume model
    deadVolumeFile = load(parameterPath)["deadVolumeFile"]
    # deadVolumeFile = ['deadVolumeCharacteristics_20210810_1653_eddec53.npz'] # MS OLD
    # deadVolumeFile = ['deadVolumeCharacteristics_20220726_0235_e81a19e.npz'] # MS LV
    # deadVolumeFile = ['deadVolumeCharacteristics_20220714_2133_6072a85.npz'] # MS LV 2
    # deadVolumeFile = ['deadVolumeCharacteristics_20230214_2002_7e5a5aa.npz'] # MS LV 3
    # deadVolumeFile = ['deadVolumeCharacteristics_20220712_1444_e81a19e.npz'] # MS LV BEST
    
    # Dead volume model
    # deadVolumeFile = [[['deadVolumeCharacteristics_20230220_1813_7e5a5aa.npz', #lowflow
    #                    'deadVolumeCharacteristics_20230220_1752_7e5a5aa.npz']],
    #                   [['deadVolumeCharacteristics_20230220_1813_7e5a5aa.npz', #lowflow
    #                    'deadVolumeCharacteristics_20230220_1752_7e5a5aa.npz']]] #highflow ZIF 8 OLD
    
    # Dead volume model
    # deadVolumeFile = [[['deadVolumeCharacteristics_20230821_1646_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_1724_b571c46.npz']],
    #                   [['deadVolumeCharacteristics_20230821_1649_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_1744_b571c46.npz']]] #highflow ZIF 8 NEW
    
    # # # Dead volume model
    # deadVolumeFile = [[['deadVolumeCharacteristics_20230821_1803_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_1849_b571c46.npz']],
    #                   [['deadVolumeCharacteristics_20230821_1813_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_1909_b571c46.npz']]] #highflow CMS Ar
    
    # Dead volume model
    # deadVolumeFile = [[['deadVolumeCharacteristics_20230821_1928_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_2018_b571c46.npz']],
    #                   [['deadVolumeCharacteristics_20230821_1937_b571c46.npz', #lowflow
    #                     'deadVolumeCharacteristics_20230821_2042_b571c46.npz']]] #highflow CMS He DO NOT USE
    
    
    # deadVolumeFile = [[['deadVolumeCharacteristics_20230321_1036_59cc206.npz', # 50A
    #                   'deadVolumeCharacteristics_20230321_1245_59cc206.npz']], # 51A
    #                   [['deadVolumeCharacteristics_20230321_1137_59cc206.npz', # 50B
    #                   'deadVolumeCharacteristics_20230321_1252_59cc206.npz']]] # 51B ZEOLITE Y CMS
    
    # deadVolumeFile = [[['deadVolumeCharacteristics_20230321_1036_59cc206.npz', # 50A
    #                   'deadVolumeCharacteristics_20230321_1252_59cc206.npz']], # 51A
    #                   [['deadVolumeCharacteristics_20230321_1137_59cc206.npz', # 50B
    #                   'deadVolumeCharacteristics_20230321_1252_59cc206.npz']]] # 51B ZEOLITE Y CMS    
    # deadVolumeFile = [[['deadVolumeCharacteristics_20220726_0235_e81a19e.npz', # 
    #                 'deadVolumeCharacteristics_20220726_0235_e81a19e.npz']], # 
    #               [['deadVolumeCharacteristics_20220726_0235_e81a19e.npz', # 
    #                 'deadVolumeCharacteristics_20220726_0235_e81a19e.npz']]] # AC
    # Isotherm parameter reference
    parameterReference = load(parameterPath)["parameterReference"]
    paramIso = load(parameterPath)["paramIso"]
    # parameterReference = [1000,1000]
    # Load the model
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    modelNonDim = modelOutputTemp[()]["variable"] 
    # This was added on 12.06 (not back compatible for error computation)
    downsampleData = load(parameterPath)["downsampleFlag"]
    ##############
    # downsampleData = True
    ##############
    # modelType = 'Kinetic'    
    # modelType = 'KineticSB'
    # modelType = 'KineticMacro'
    modelType = load(parameterPath)["modelType"]

    print("Objective Function",round(modelOutputTemp[()]["function"],0))

    numPointsExp = np.zeros(len(fileName))
    for ii in range(len(fileName)): 
        fileToLoad = fileName[ii]
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
    
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)
    ##############
    ##############
    downsampleInt = numPointsExp/numPointsExp
    ##############
    ##############
    # Multiply the paremeters by the reference values
    x = np.multiply(modelNonDim,parameterReference)
    # x = np.zeros(6)
    x = np.zeros(8)
    x = np.zeros(9)
    
    if modelType == 'KineticSBMacro':
        x[0:6] = paramIso[0:-3]
        x[-3:] = np.multiply(modelNonDim,parameterReference)   
    elif modelType == 'KineticMacro':
        x[0:6] = paramIso[0:-3]
        x[-3:] = np.multiply(modelNonDim,parameterReference)      
    else:
        x[0:6] = paramIso[0:-2]
        x[-2:] = np.multiply(modelNonDim,parameterReference)    
    # x = np.zeros(5)
    

    # x[0:4] = [20, 5.51151743793683e-07, 19272.6327577962,1.12005137776376]# MT SSS  
    # x[-2:] = [0.2445, 983] # MT HIGH FLOW
    # x[-2:] = [0.294, 23.9] # MT LOW FLOW   
    # x[-2:] = [90, 35] # MT ALL FLOW 
    # x[-2:] =   [0.58608418, 60,]
    
    # x[0:4] = [6.51504325937662, 1.11248296208017e-07, 24942.1235836143,1.06047499753496]# MCB30 SSS
    # x[-2:] =  [0.49753, 3.09429] # MCB30 SSS HIGH FLOW    
    # x[-2:] = [0.182, 12.7568] # MCB30 SSS HIGH FLOW
    # x[-2:] = [0.0871, 990] # MCB30 SSS LOW FLOW
    # x[-2:] = [1.09207, 2.9566] # MCB30 SSS ALL FLOW
    
    # x[0:4] = [9.6369, 2.7438e-7, 2.1386e+04,0.9893]# MCB20 SSS
    # x[-2:] = [0.333, 4.00] # MCB20 SSS HIGH FLOW
    # x[-2:] = [0.10933, 0.873] # MCB20 SSS LOW FLOW
    # x[-2:] = [0.22544, 1.2982] # MCB20 SSS ALL FLOW
    # x[-2:] = [382, 0.637] # MCB20 SSS ALL FLOW HIGH COMP ONLY 
    # x[-2:] = [382, 0.637] # MCB20 SSS ALL FLOW HIGH COMP ONLY 

    # x[-2:] = [9.89720284e-02, 9.56339260e-01] # MCB20 SSS LOW FLOW
    # x[-2:] = [0.085, 9.56339260e-01] # MCB20 SSS LOW 

    # x[0:6] = [4.3418e-01, 1.0555e-06 , 3.2322e+04, 6.6351e+00, 2.0803e-07, 2.6108e+04] # ZYH DSL 1 bara new
    # x[0:6] = [6.4975e+00, 3.5355e-07, 3.1108e+04, 9.0420e-01, 5.1101e-05, 2.3491e+04] # ZYNa DSL 1 bara new
    # x[0:6] = [5.1394e+00, 2.7760e-07, 2.8886e+04, 2.6934e+00, 1.2966e-06, 2.9451e+04] # ZYTMA DSL 1 bara new
    # x[0:6] = [9.7667e-01, 1.0749e-05, 2.4312e+04, 3.4760e+00, 2.2254e-07, 2.7095e+04]  # CMS3K DSL 1 bara new
# 
    # x[-3:] = [0.07748405*1e3, 0.01526513*1e3, 0.07375*1e3 ] # SBmacro run 1 best ZYH
     # x[-2:] =   [0.6e-3*15, 0,] # CMS3K 
    # x[-2:] = [0*1e3, 0.007188509*1e3,] # CMS3K 
    # x[-3:] = [0.17716856*1e3, 0.02432568*1e3, 0] # SB run 1 best Helium
    # x[-3:] = [0.2125*1e3, 0.0246389*1e3, 0] # SB run 1 best Ar

    # x[-3:] = [0.03530858*1e3, 0.01617368*1e3, 0.00605428*1e3] # SBmacro run 1 best He
    # x[-3:] = [1.48859751e+02*1e3, 3.63321115e-02*1e3, 8.06733749e-01*1e3] # SBmacro run 1 best ZYH He
    # x[-3:] = [150*1e3, 0.037*1e3, 0.2*1e3 ] # SBmacro run 1 best ZYH
    # x[-3:] = [8.50392278e+03, 3.19273655e+01,  9.99707560e+02]
    # x[-3:] = [150e+03, 3.79273655e+01,  9.99707560e+02]
    # x[-3:] = [0.25*1e3, 0.0250343*1e3,  0]
    # x[-3:] = [2.44436243e+05, 3.79187820e+01, 300.08352988e+01]
    # x[-3:] = [2.49884380e+05, 3.96945399e+01,0]
    # x[-2:] = [0.00152662*1000, 0.01235128*1000,]
    computedError = 0
    numPoints = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])
    massBalanceALL = np.zeros((len(fileName),3))
    massBalanceEXP = np.zeros((len(fileName),3))
    massBalanceDV = np.zeros((len(fileName),3))
    
    # Create the instance for the plotsx
    
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
        # if ii < 3:
        #     timeElapsedExp = timeElapsedExp-0.2
        # timeElapsedExp[timeElapsedExp < 0] = 0
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

        if simulateModel:
            # Parse out parameter values
            if modelType == 'KineticSBMacro':
                isothermModel = x[0:-3]
                rateConstant_1 = x[-3]
                rateConstant_2 = x[-2]
                rateConstant_3 = x[-1]     
            elif modelType == 'KineticMacro':
                isothermModel = x[0:-3]
                rateConstant_1 = x[-3]
                rateConstant_2 = x[-2]
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
                                                        temperature = temperatureExp[ii],
                                                        adsorbentDensity = adsorbentDensity,
                                                        modelType = modelType)
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

            # Compute the mass balance at the end of the ZLC
            massBalanceALL[ii,0] = moleFracExp[0]
            massBalanceALL[ii,1] = ((np.trapz(resultMat[0,:],np.multiply(resultMat[3,:],timeElapsedExp))
                                    - volGas*moleFracExp[0])*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000))
            massBalanceALL[ii,2] = volGas*moleFracExp[0]*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000)   
        
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
            moleFracDV[moleFracDV<0.01] = 0
            # moleFracDV[moleFracDV<0.0001] = 0
            # Compute the mass balance at the end end of the ZLC for experiments
            massBalanceEXP[ii,0] = moleFracExp[0]
            massBalanceEXP[ii,1] = ((np.trapz(moleFracExp,np.multiply(flowRateExp*1e-6,timeElapsedExp))- 
                                     np.trapz(moleFracDV,np.multiply(flowInDV*1e-6,timeElapsedExp))+volSorbent)
                                    *(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000))
            massBalanceEXP[ii,2] = ((np.trapz(moleFracExp,np.multiply(flowRateExp*1e-6,timeElapsedExp))
                                     - deadVolumeIntegral[ii]+volSorbent+
                                     volGas*moleFracExp[0])*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000))
            # massBalanceEXP[ii,1] =((np.trapz(timeElapsedExp,np.multiply(moleFracExp,flowRateExp*1e-6))
            #                           - np.trapz(timeElapsedExp,np.multiply(moleFracDV,flowInDV))+volSorbent)*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000))
            # massBalanceEXP[ii,2] = volGas*moleFracExp[0]*(pressureTotal/(8.314*temperatureExp[ii]))/(massSorbent/1000) 
        # y - Linear scale
        ax1.semilogy(timeElapsedExp,moleFracExp, markersize = 2,
                marker = markerForPlot[ii],linewidth = 1.5,
                      alpha=0.2,markeredgecolor=colorsForPlot[ii],markeredgewidth=0.1,markerfacecolor='none') # Experimental response
        if simulateModel:
            if useFlow:
                legendStr = str(round(np.mean(flowRateExp),2))+" ccs"
            else:
                legendStr = str(temperatureExp[ii])+" K"
            ax1.plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],label=legendStr,alpha = 1) # Simulation response    
            # if ii==len(fileName)-1:
            ax1.plot(timeElapsedExp,moleFracDV,
                          color='#118ab2',label="DV",alpha=0.1,
                          linestyle = '-') # Dead volume simulation response    

        ax1.set(xlabel='$t$ [s]', 
                ylabel='$y_1$ [-]',
                # xlim = [0,250], ylim = [1e-2, 1])    
                # xlim = [0,400], ylim = [1e-2, 1])    
                # xlim = [0,1000], ylim =  [1e-2, 1])   
                # xlim = [0,600], ylim =  [1e-2, 1])   
                ylim =  [1e-2, 1])   
        ax1.locator_params(axis="x", nbins=4)
        ax1.autoscale(enable=None, axis="x", tight=False)

        # ax1.legend()

        # Ft - Log scale        
        ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp),moleFracExp,
                      marker = markerForPlot[ii],linewidth = 0,
                      alpha=0.4,markeredgecolor=colorsForPlot[ii],markeredgewidth=0.1,markerfacecolor='none') # Experimental response
        if simulateModel:
            ax2.semilogy(np.multiply(resultMat[3,:]*1e6,timeElapsedExp),moleFracSim,
                          color=colorsForPlot[ii],label=str(round(np.mean(resultMat[3,:]*1e6),2))+" ccs",
                          alpha = 1) # Simulation response
            ax2.plot(np.multiply(timeElapsedExp,flowInDV*1e6),moleFracDV,
                          color='#118ab2',label="DV",alpha=0.1,
                          linestyle = '-') # Dead volume simulation response  
        ax2.set(xlabel='$Ft$ [cc]', 
                # xlim = [0,90], ylim = [1e-2, 1])   
                # xlim = [0,50], ylim = [1e-2, 1])   
                # xlim = [0,250], ylim = [1e-2, 1])   
                # xlim = [0,130], ylim = [1e-2, 1])   

                # xlim = [0,2000], ylim = [1e-4, 0.021])  
                ylim =  [1e-2, 1])   
        ax1.locator_params(axis="x", nbins=4)

        ax2.locator_params(axis="x", nbins=4)
        
        # Flow rates
        ax3.plot(timeElapsedExp,flowRateExp,
                marker = markerForPlot[ii],linewidth = 0,
                alpha=0.8,markeredgecolor=colorsForPlot[ii],markeredgewidth=0.1,markerfacecolor='none',label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
        if simulateModel:
            ax3.plot(timeElapsedExp,resultMat[3,:]*1e6,
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
        
    plt.show()
    
    # Print the MLE error
    if simulateModel:
        computedError = computeMLEError(moleFracExpALL,moleFracSimALL, 
                                        downsampleData = downsampleData,)
        print("Sanity check objective function: ",round(computedError,0))
    
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