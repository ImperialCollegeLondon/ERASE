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
# Plots for the ZLC response using experimental data
#
# Last modified:
# - 2023-01-31, HA: Initial creation
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
from simulateCombinedModel import simulateCombinedModel
import pdb
from scipy.optimize import fsolve 
from scipy.optimize import leastsq 
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
fileParameter = 'zlcParameters_20221208_1241_7e5a5aa.npz' # ZIF8 MT EXP
# fileParameter = 'zlcParameters_20210824_1522_c8173b1.npz'

# Flag to plot simulations
simulateModel = False

# Flag to plot dead volume results
plotFt = False
    
# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);

# Plot colors
colorsForPlot = ["#faa307","#d00000","#03071e"]*4
markerForPlot = ["o"]*20


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
#                     'ZLC_ActivatedCarbon_Sim03A_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim05A_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim01B_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim03B_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim05B_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim02A_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim04A_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim06A_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim02B_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim04B_Output.mat',
#                     'ZLC_ActivatedCarbon_Sim06B_Output.mat',]

# rawFileName = ['ZLC_ActivatedCarbon_Exp76B_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp74B_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp72B_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp77B_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp75B_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp73B_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp76A_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp74A_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp72A_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp77A_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp75A_Output.mat',
#                        'ZLC_ActivatedCarbon_Exp73A_Output.mat',]

# rawFileName = ['ZLC_ActivatedCarbon_Sim05B_Output.mat',]


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
#                     'ZLC_Zeolite13X_Sim03A_Output.mat',
#                     'ZLC_Zeolite13X_Sim05A_Output.mat',
#                     'ZLC_Zeolite13X_Sim01B_Output.mat',
#                     'ZLC_Zeolite13X_Sim03B_Output.mat',
#                     'ZLC_Zeolite13X_Sim05B_Output.mat',
#                     'ZLC_Zeolite13X_Sim02A_Output.mat',
#                     'ZLC_Zeolite13X_Sim04A_Output.mat',
#                     'ZLC_Zeolite13X_Sim06A_Output.mat',
#                     'ZLC_Zeolite13X_Sim02B_Output.mat',
#                     'ZLC_Zeolite13X_Sim04B_Output.mat',
#                     'ZLC_Zeolite13X_Sim06B_Output.mat',]
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
rawFileName = ['ZLC_ZIF8_MT_Exp01A_Output.mat',
            'ZLC_ZIF8_MT_Exp03A_Output.mat',
            'ZLC_ZIF8_MT_Exp07A_Output.mat',
            'ZLC_ZIF8_MT_Exp01B_Output.mat',
            'ZLC_ZIF8_MT_Exp03B_Output.mat',
            'ZLC_ZIF8_MT_Exp07B_Output.mat',
            'ZLC_ZIF8_MT_Exp02A_Output.mat',
            'ZLC_ZIF8_MT_Exp04A_Output.mat',
            'ZLC_ZIF8_MT_Exp08A_Output.mat',
            'ZLC_ZIF8_MT_Exp02B_Output.mat',
            'ZLC_ZIF8_MT_Exp04B_Output.mat',
            'ZLC_ZIF8_MT_Exp08B_Output.mat',]
    
# rawFileName = ['ZLC_ZIF8_MCB30_Exp01A_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp03A_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp07A_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp01B_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp03B_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp07B_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp02A_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp04A_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp08A_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp02B_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp04B_Output.mat',
#             'ZLC_ZIF8_MCB30_Exp08B_Output.mat',]
# rawFileName = ['ZLC_ZIF8_MCB30_Exp01A_Output.mat',]

# rawFileName = ['ZLC_ZIF8_MCB20_Exp01A_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp03A_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp07A_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp01B_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp03B_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp07B_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp02A_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp04A_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp08A_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp02B_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp04B_Output.mat',
#             'ZLC_ZIF8_MCB20_Exp08B_Output.mat',]
    
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

# rawFileName = ['ZLC_Lewatit_DA_Exp05A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp07A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp09A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp11A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp13A_Output.mat',
#                     'ZLC_Lewatit_DA_Exp05B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp07B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp09B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp11B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp13B_Output.mat',]

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
# rawFileNameDV = ['ZLC_Empty_Exp10B_Output.mat',]
# rawFileNameDV = ['ZLC_Empty_Exp14A_Output.mat',
#                  'ZLC_Empty_Exp14A_Output.mat',
#                  'ZLC_Empty_Exp14A_Output.mat',
#                  'ZLC_Empty_Exp14B_Output.mat',
#                  'ZLC_Empty_Exp14B_Output.mat',
#                  'ZLC_Empty_Exp14B_Output.mat',
#                  'ZLC_Empty_Exp17A_Output.mat',
#                  'ZLC_Empty_Exp17A_Output.mat',
#                  'ZLC_Empty_Exp17A_Output.mat',
#                  'ZLC_Empty_Exp17B_Output.mat',
#                  'ZLC_Empty_Exp17B_Output.mat',
#                  'ZLC_Empty_Exp17B_Output.mat',]

# Generate .npz file for python processing of the .mat file 
filesToProcess(True,os.path.join('..','experimental','runData'),rawFileNameDV,'DV')
    
# ZLC parameter model path
parameterPath = os.path.join('..','simulationResults',fileParameter)

# Temperature (for each experiment)
# temperatureExp = [344.69, 325.39, 306.15]*4 # AC Experiments
# temperatureExp = [308.15, 328.15, 348.15]*4 # AC Simulations
# temperatureExp = [344.6, 325.49, 306.17,]*4 # BN (2 pellets) Experiments
# temperatureExp = [308.15, 328.15, 348.15]*4 # BN (2 pellets) Simulations 
temperatureExp = [ 303.15, 293.15, 283.15, ]*4 # ZIF8 
# temperatureExp = [ 283.15, 293.15, 303.15,]*4 # BNFAS
# temperatureExp = [ 363.15, 348.15, 333.15, 318.15, 303.15,]*2 # lewatit

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
## Adsorbent properties
# Adsorbent density [kg/m3]
# This has to be the skeletal density
# adsorbentDensity = 1680 # Activated carbon skeletal density [kg/m3]
# adsorbentDensity = 4100 # Zeolite 13X H 
# adsorbentDensity = 1250 # BNFASp skeletal density [kg/m3]
# adsorbentDensity = 2320 # BNpFAS skeletal density [kg/m3]
# adsorbentDensity = 988 # Lewatit skeletal density [kg/m3]
adsorbentDensity = 1555 # ZIF-8 MT
# adsorbentDensity = 2400 # ZIF-8 MCB20
# adsorbentDensity = 2100 # ZIF-8 MCB30


# Particle porosity
# particleEpsilon = 0.61 # AC
# particleEpsilon = 0.79 # Zeolite 13X H
# particleEpsilon = 0.64 # BNFASp
# particleEpsilon = 0.44 # Lewatit
# particleEpsilon = 0.67 # BNpFAS
particleEpsilon = 0.43 # ZIF-8 MT
# particleEpsilon = 0.62 # ZIF-8 MCB20
# particleEpsilon = 0.59 # ZIF-8 MCB30

# Particle mass [g]
# massSorbent = 0.0625  # AC
# massSorbent = 0.0594 # Zeolite 13X H
# massSorbent = 0.1  # BNFASp
# massSorbent = 0.1  # BNFASp
# massSorbent = 0.0262  # Lewatit
# massSorbent = 0.12  # BNpFAS
massSorbent = 0.059 # ZIF-8 MT
# massSorbent = 0.0988 # ZIF-8 MCB20
# massSorbent = 0.102 # ZIF-8 MCB30
# Volume of sorbent material [m3]
volSorbent = (massSorbent/1000)/adsorbentDensity

# Volume of gas chamber (dead volume) [m3]
volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

# Dead volume model
deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
deadVolumeFile = 'deadVolumeCharacteristics_20220712_1444_e81a19e.npz' # MS OLD
# deadVolumeFile = 'deadVolumeCharacteristics_20220726_0235_e81a19e.npz' # MS LV
# deadVolumeFile = 'deadVolumeCharacteristics_20220823_1542_e81a19e.npz' # DA LV
# Isotherm parameter reference
parameterReference = load(parameterPath)["parameterReference"]
# Load the model
modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
modelNonDim = modelOutputTemp[()]["variable"] 
# modelNonDim =[0.0714112 , 0.14472916, 0.66973443, 0.14185384 ,0.04062939] # ZIF8 MT

# This was added on 12.06 (not back compatible for error computation)
downsampleData = load(parameterPath)["downsampleFlag"]
##############
##############
downsampleData = False
##############
##############
print("Objective Function",round(modelOutputTemp[()]["function"],0))
fileNameDV = filesToProcess(False,[],[],'DV')

numPointsExp = np.zeros(len(fileName))
numPointsExpDV = np.zeros(len(fileNameDV))

# Get the processed file names
deadVolumeIntegral = np.zeros(len(fileNameDV))
for ii in range(len(fileName)): 
    fileToLoad = fileName[ii]
    # Load experimental molefraction
    timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
    numPointsExp[ii] = len(timeElapsedExp)
    fileToLoadDV = fileNameDV[ii]
    # Load experimental molefraction
    timeElapsedExpDV = load(fileToLoadDV)["timeElapsed"].flatten()
    numPointsExpDV[ii] = len(timeElapsedExpDV)
# Downsample intervals
downsampleInt = numPointsExp/np.min(numPointsExp)
downsampleIntDV = numPointsExpDV/np.min(numPointsExpDV)

##############
##############
# downsampleInt = numPointsExp/numPointsExp
##############
##############
# Multiply the paremeters by the reference values
x = np.multiply(modelNonDim,parameterReference)
# x[0:3] = [2.30285,2.62376e-08,31701.8]
x[0:3] = [2.99999956e+01, 9.62292726e-08, 2.15988572e+04,]# MT
x[3:5] = [1.70e-01, 9.83e+02] # MT
# x[3:5] = [1.02, 16.79] # AC
# x[0:3] = [7.9340, 3.4276e-07, 2.1424e+04,]# MCB20
# x[3:5] = [1.15726, 14.29728]
# x[3:5] = [0.419, 995]
# x[3:5] = [2.37651, 1.0022]


# Initialize loadings
computedError = 0
numPoints = 0
moleFracExpALL = np.array([])
moleFracSimALL = np.array([])
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
    ####### DV EXPERIMENTS #######
    # Initialize outputs
    moleFracSim = []  
    # Load experimental time, molefraction and flowrate (accounting for downsampling)
    timeElapsedExpTemp2 = load(fileToLoad)["timeElapsed"].flatten()
    moleFracExpTemp2 = load(fileToLoad)["moleFrac"].flatten()
    flowRateTemp2 = load(fileToLoad)["flowRate"].flatten()
    timeElapsedExp = timeElapsedExpTemp2
    moleFracExp = moleFracExpTemp2
    normMoleFracExp = 1-moleFracExp/moleFracExp[0]
    normMoleFracExp[normMoleFracExp<0] = 0
    normMoleFracExp = np.sort(normMoleFracExp)
    flowRateExp = flowRateTemp2
    volBulk = (volSorbent+volGas)
    timeSorbVolume = volBulk/(flowRateExp[-1]*1e-6)  
    # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
    timeInt = np.linspace(0,timeElapsedExp[-1],1000)
    flowInDV = np.zeros(len(timeInt))          
    flowInDV[:] = np.mean(flowRateExp[-1:-3:-1]*1e-6)
    # Call the deadVolume Wrapper function to obtain the outlet mole fraction
    deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
    modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
    pDV = modelOutputTemp[()]["variable"]
    dvFileLoadTemp = load(deadVolumePath)
    flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
    msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
    ####### DV EXPERIMENTS #######
    # pdb.set_trace()
    moleFracDV = deadVolumeWrapper(timeInt, flowInDV*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
    # moleFracDV[moleFracDV<0.01] = 0
    timeIntDV = timeInt[moleFracDV>0.01]
    moleFracDV = moleFracDV[moleFracDV>0.01]
    
    # Initialize outputs
    moleFracSimDV = []
    # Path of the file name
    fileToLoadDV = fileNameDV[ii]   
    # yVals = np.linspace(1-np.max(moleFracExp),0.99,1000)
    yVals = np.linspace(1-moleFracExp[0],0.99,100)
    # Load experimental time, molefraction and flowrate (accounting for downsampling)
    timeElapsedExpTemp = load(fileToLoadDV)["timeElapsed"].flatten()
    moleFracExpTemp = load(fileToLoadDV)["moleFrac"].flatten()
    flowRateTemp = load(fileToLoadDV)["flowRate"].flatten()
    # normMoleFracExpTemp = 1-moleFracExpTemp/moleFracExpTemp[0]
    # normMoleFracExpTemp[normMoleFracExpTemp<0] = 0    
    # normMoleFracExpTemp = np.sort(normMoleFracExpTemp)
    timeElapsedExpDV = np.interp(yVals,1-moleFracDV,timeIntDV)
    timeElapsedExp = np.interp(yVals,1-moleFracExp,timeElapsedExp)
    flowRateExp = np.interp(timeElapsedExp,1-moleFracExpTemp,flowRateTemp)
    # flowRateExp[:] = flowRateExp[-1]
    moleFracExp = 1-yVals
    # timeElapsedExpDV = np.sort(timeElapsedExpDV) + 5.4
    # moleFractExpInterp = np.interp(normMoleFracExp,1-moleFracDV/moleFracDV[0],normMoleFracExp)

    # timeElapsedExpDES = np.interp(1-moleFracDV,np.around(normMoleFracExp,3),timeInt)
    # timeElapsedExpDV = np.interp(1-moleFracExp/moleFracExp[0],1-moleFracExpTemp/moleFracExpTemp[0],timeElapsedExpTemp)
      
    # timeElapsedExpDV = timeElapsedExpTemp[::int(np.round(downsampleIntDV[ii]))]
    moleFracExpDV = moleFracExpTemp[::int(np.round(downsampleIntDV[ii]))]
    flowRateExpDV = flowRateTemp[::int(np.round(downsampleIntDV[ii]))]
            
    # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
    timeInt = timeElapsedExp
    timeSorb =  np.zeros(len(timeElapsedExp))
    timeSorb[:]  = timeSorbVolume
    # Print experimental volume 
    # print("Experiment",str(ii+1),round(np.trapz(moleFracExp,np.multiply(flowRateExp,timeElapsedExp)),2))

    # y - Linear scale
    ax1.semilogy(timeElapsedExp-timeElapsedExpDV+timeSorb,moleFracExp,
            # marker = 'markerForPlot[ii]',linewidth = 1,
            marker = '',linewidth = 1,
            color=colorsForPlot[ii],alpha=1) # Experimental response
    
    ax1.set(xlabel='$t$ [s]', 
            ylabel='$y_1$ [-]',
            xlim = [-1,70], ylim = [1e-2, 1])    
            # xlim = [0,8000], ylim = [1e-4, 0.021])  
    ax1.locator_params(axis="x", nbins=4)
    # ax1.legend()

    # Ft - Log scale        
    ax2.semilogy(np.multiply(flowRateExp,timeElapsedExp-timeElapsedExpDV+timeSorb),moleFracExp,
                  # marker = markerForPlot[ii],linewidth = 0,
                  marker = '',linewidth = 1,
                  color=colorsForPlot[ii],alpha=1) # Experimental response
   
    ax2.set(xlabel='$Ft$ [cc]', 
            xlim = [-1,20], ylim = [1e-2, 1])   
            # xlim = [0,2000], ylim = [1e-4, 0.021])  

    ax2.locator_params(axis="x", nbins=4)
    
    # Flow rates
    ax3.plot(timeElapsedExp-timeElapsedExpDV+timeSorb,flowRateExp,
            marker = markerForPlot[ii],linewidth = 0,
            color=colorsForPlot[ii],alpha=0.1,label=str(round(np.mean(flowRateExp),2))+" ccs") # Experimental response
   
    ax3.set(xlabel='$t$ [s]', 
            ylabel='$F$ [ccs]',
            xlim = [-1,10], ylim = [0, 3])
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

# root = fsolve(ZLCObjFun, timeElapsedExpTemp2 ,args = [moleFracExpTemp2,
#               timeElapsedExpTemp2,
#               flowRateTemp2,
#               deadVolumeFile])
# root
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
    
def ZLCObjFun(x,args):
    from deadVolumeWrapper import deadVolumeWrapper
    import os
    from numpy import load
    import numpy as np
    import pdb
    
    
    moleFracExpTemp2 = args[0]
    timeElapsedExpTemp2 = args[1]
    flowRateTemp2 = args[2]
    deadVolumeFile = args[3]
    
    flowRateExpTemp = np.zeros(len(flowRateTemp2))
    flowRateExpTemp[:] = flowRateTemp2[-1]
    flowRateTemp2 = flowRateExpTemp
    deadVolumeDir = '..' + os.path.sep + 'simulationResults/'
    modelOutputTemp = load(deadVolumeDir+deadVolumeFile, allow_pickle=True)["modelOutput"]
    deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
    # Parse out dead volume parameters
    DVvars = modelOutputTemp[()]["variable"]
    dvFileLoadTemp = load(deadVolumePath)
    flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
    msDeadVolumeFile = str(dvFileLoadTemp["msDeadVolumeFile"])
    print(x)
    # pdb.set_trace()
    # Call the deadVolume Wrapper function to obtain the outlet mole fraction
    return abs(moleFracExpTemp2 - deadVolumeWrapper(abs(np.sort(x)), flowRateTemp2, DVvars, flagMSDeadVolume, msDeadVolumeFile,
                                    initMoleFrac = moleFracExpTemp2[0], feedMoleFrac = moleFracExpTemp2))