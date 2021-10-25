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
# Plots for the comparison of isotherms obtained from different devices and 
# different fits from ZLC
#
# Last modified:
# - 2021-08-20, AK: Introduce macropore diffusivity (for sanity check)
# - 2021-08-20, AK: Change definition of rate constants
# - 2021-07-01, AK: Cosmetic changes
# - 2021-06-15, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
from computeEquilibriumLoading import computeEquilibriumLoading
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from numpy import load
import auxiliaryFunctions
plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Save flag and file name extension
saveFlag = False
saveFileExtension = ".png"

# Colors
colorForPlot = ["faa307","d00000","03071e"]
# colorForPlot = ["E5383B","6C757D"]

# Plot text
plotText = 'DSL'

# Universal gas constant
Rg = 8.314

# Total pressure
pressureTotal = np.array([1.e5]);

# Define temperature
temperature = [308.15, 328.15, 348.15]

# CO2 molecular diffusivity
molDiffusivity = 1.6e-5 # m2/s

# Particle Tortuosity
tortuosity = 3

# Particle Radius
particleRadius = 2e-3

# AC Isotherm parameters
x_VOL = [4.65e-1, 1.02e-5 , 2.51e4, 6.51, 3.51e-7, 2.57e4] # (Hassan, QC)

# 13X Isotherm parameters (L pellet)
# x_VOL = [2.50, 2.05e-7, 4.29e4, 4.32, 3.06e-7, 3.10e4] # (Hassan, QC)
# x_VOL = [3.83, 1.33e-08, 40.0e3, 2.57, 4.88e-06, 35.16e3] # (Hassan, QC - Bound delU to 40e3)

# BN Isotherm parameters
# x_VOL = [7.01, 2.32e-07, 2.49e4, 0, 0, 0] # (Hassan, QC)

# ZLC Parameter estimates
# New kinetic model
# Both k1 and k2 present

# Activated Carbon Experiments
# zlcFileName = ['zlcParameters_20210822_0926_c8173b1.npz',
#                 'zlcParameters_20210822_1733_c8173b1.npz',
#                 'zlcParameters_20210823_0133_c8173b1.npz',
#                 'zlcParameters_20210823_1007_c8173b1.npz',
#                 'zlcParameters_20210823_1810_c8173b1.npz']

# Activated Carbon Experiments - dqbydc = Henry's constant
# zlcFileName = ['zlcParameters_20211002_0057_c8173b1.npz',
#                 'zlcParameters_20211002_0609_c8173b1.npz',
#                 'zlcParameters_20211002_1119_c8173b1.npz',
#                 'zlcParameters_20211002_1638_c8173b1.npz',
#                 'zlcParameters_20211002_2156_c8173b1.npz']

# Activated Carbon Experiments - Dead volume
# zlcFileName = ['zlcParameters_20211011_1334_c8173b1.npz',
#                 'zlcParameters_20211011_2058_c8173b1.npz',
#                 'zlcParameters_20211012_0437_c8173b1.npz',
#                 'zlcParameters_20211012_1247_c8173b1.npz',
                # 'zlcParameters_20211012_2024_c8173b1.npz']

# Activated Carbon Simulations (Main)
zlcFileName = ['zlcParameters_20210823_1104_03c82f4.npz',
                'zlcParameters_20210824_0000_03c82f4.npz',
                'zlcParameters_20210824_1227_03c82f4.npz',
                'zlcParameters_20210825_0017_03c82f4.npz',
                'zlcParameters_20210825_1151_03c82f4.npz']

# Activated Carbon Simulations (Effect of porosity)
# 0.90
# zlcFileName = ['zlcParameters_20210922_2242_c8173b1.npz',
#                 'zlcParameters_20210923_0813_c8173b1.npz',
#                 'zlcParameters_20210923_1807_c8173b1.npz',
#                 'zlcParameters_20210924_0337_c8173b1.npz',
#                 'zlcParameters_20210924_1314_c8173b1.npz']

# 0.35
# zlcFileName = ['zlcParameters_20210923_0816_c8173b1.npz',
#                 'zlcParameters_20210923_2040_c8173b1.npz',
#                 'zlcParameters_20210924_0952_c8173b1.npz',
#                 'zlcParameters_20210924_2351_c8173b1.npz',
#                 'zlcParameters_20210925_1243_c8173b1.npz']

# Activated Carbon Simulations (Effect of mass)
# 1.05
# zlcFileName = ['zlcParameters_20210925_1104_c8173b1.npz',
#                 'zlcParameters_20210925_2332_c8173b1.npz',
#                 'zlcParameters_20210926_1132_c8173b1.npz',
#                 'zlcParameters_20210926_2248_c8173b1.npz',
#                 'zlcParameters_20210927_0938_c8173b1.npz']

# 0.95
# zlcFileName = ['zlcParameters_20210926_2111_c8173b1.npz',
#                 'zlcParameters_20210927_0817_c8173b1.npz',
#                 'zlcParameters_20210927_1933_c8173b1.npz',
#                 'zlcParameters_20210928_0647_c8173b1.npz',
#                 'zlcParameters_20210928_1809_c8173b1.npz']

# Activated Carbon Simulations (Effect of dead volume)
# TIS + MS
# zlcFileName = ['zlcParameters_20211015_0957_c8173b1.npz',
#                 'zlcParameters_20211015_1744_c8173b1.npz',
#                 'zlcParameters_20211016_0148_c8173b1.npz',
#                 'zlcParameters_20211016_0917_c8173b1.npz',
#                 'zlcParameters_20211016_1654_c8173b1.npz']

# Boron Nitride Experiments
# zlcFileName = ['zlcParameters_20210823_1731_c8173b1.npz',
#                 'zlcParameters_20210824_0034_c8173b1.npz',
#                 'zlcParameters_20210824_0805_c8173b1.npz',
#                 'zlcParameters_20210824_1522_c8173b1.npz',
#                 'zlcParameters_20210824_2238_c8173b1.npz',]

# Boron Nitride Simulations
# zlcFileName = ['zlcParameters_20210823_1907_03c82f4.npz',
#                 'zlcParameters_20210824_0555_03c82f4.npz',
#                 'zlcParameters_20210824_2105_03c82f4.npz',
#                 'zlcParameters_20210825_0833_03c82f4.npz',
#                 'zlcParameters_20210825_2214_03c82f4.npz']

# Zeolite 13X Simulations
# zlcFileName = ['zlcParameters_20210824_1102_c8173b1.npz',
#                 'zlcParameters_20210825_0243_c8173b1.npz',
#                 'zlcParameters_20210825_1758_c8173b1.npz',
#                 'zlcParameters_20210826_1022_c8173b1.npz',
#                 'zlcParameters_20210827_0104_c8173b1.npz']

# Create the grid for mole fractions
y = np.linspace(0,1.,100)
# Initialize isotherms 
isoLoading_VOL = np.zeros([len(y),len(temperature)])
isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
kineticConstant_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
kineticConstant_Macro = np.zeros([len(zlcFileName),len(y),len(temperature)])
objectiveFunction = np.zeros([len(zlcFileName)])

# Loop over all the mole fractions
# Volumetric data
for jj in range(len(temperature)):
    for ii in range(len(y)):
        isoLoading_VOL[ii,jj] = computeEquilibriumLoading(isothermModel=x_VOL,
                                                          moleFrac = y[ii],
                                                          temperature = temperature[jj])
# Loop over all available ZLC files
for kk in range(len(zlcFileName)):
    # ZLC Data 
    parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
    parameterReference = load(parameterPath)["parameterReference"]
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
    modelNonDim = modelOutputTemp[()]["variable"] 
    # Multiply the paremeters by the reference values
    x_ZLC = np.multiply(modelNonDim,parameterReference)
    print(x_ZLC)

    adsorbentDensity = load(parameterPath, allow_pickle=True)["adsorbentDensity"]
    particleEpsilon = load(parameterPath)["particleEpsilon"]

    # Print names of files used for the parameter estimation (sanity check)
    fileNameList = load(parameterPath, allow_pickle=True)["fileName"]
    print(fileNameList)
    
    # Parse out the isotherm parameter
    isothermModel = x_ZLC[0:-2]
    rateConstant_1 = x_ZLC[-2]
    rateConstant_2 = x_ZLC[-1]

    for jj in range(len(temperature)):
        for ii in range(len(y)):
            isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                 moleFrac = y[ii], 
                                                                 temperature = temperature[jj]) # [mol/kg]
            # Partial pressure of the gas
            partialPressure = y[ii]*pressureTotal
            # delta pressure to compute gradient
            delP = 1e-3
            # Mole fraction (up)
            moleFractionUp = (partialPressure + delP)/pressureTotal
            # Compute the loading [mol/m3] @ moleFractionUp
            equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature[jj],
                                                            moleFrac=moleFractionUp,
                                                            isothermModel=isothermModel) # [mol/kg]
            
            # Compute the gradient (delq*/dc)
            dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[kk,ii,jj])*adsorbentDensity/(delP/(Rg*temperature[jj])) # [-]

            # Rate constant 1 (analogous to micropore resistance)
            k1 = rateConstant_1

            # Rate constant 2 (analogous to macropore resistance)
            k2 = rateConstant_2/dqbydc
                        
            # Overall rate constant
            # The following conditions are done for purely numerical reasons
            # If pure (analogous) macropore
            if k1<1e-12:
                rateConstant = k2
            # If pure (analogous) micropore
            elif k2<1e-12:
                rateConstant = k1
            # If both resistances are present
            else:
                rateConstant = 1/(1/k1 + 1/k2)
            
            # Rate constant (overall)
            kineticConstant_ZLC[kk,ii,jj] = rateConstant
        
            # # Macropore resistance from QC data
            # # Compute dqbydc for QC isotherm
            # equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature[jj],
            #                                     moleFrac=moleFractionUp,
            #                                     isothermModel=x_VOL) # [mol/kg]
            # dqbydc_True = (equilibriumLoadingUp-isoLoading_VOL[ii,jj])*adsorbentDensity/(delP/(Rg*temperature[jj])) # [-]

            # # Macropore resistance
            # kineticConstant_Macro[kk,ii,jj] = (15*particleEpsilon*molDiffusivity
            #                                    /(tortuosity*(particleRadius)**2)/dqbydc_True)
            
# Plot the isotherms    
fig = plt.figure
ax1 = plt.subplot(1,2,1)        
for jj in range(len(temperature)):
    ax1.plot(y,isoLoading_VOL[:,jj],color='#'+colorForPlot[jj],label=str(temperature[jj])+' K') # Ronny's isotherm
    for kk in range(len(zlcFileName)):
        ax1.plot(y,isoLoading_ZLC[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.2) # ALL

ax1.set(xlabel='$P$ [bar]', 
ylabel='$q^*$ [mol kg$^\mathregular{-1}$]',
xlim = [0,1], ylim = [0, 3]) 
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)
ax1.legend()   

# Plot the objective function
fig = plt.figure
ax2 = plt.subplot(1,2,2)       
for kk in range(len(zlcFileName)):
    ax2.scatter(kk+1,objectiveFunction[kk]) # ALL

ax2.set(xlabel='Iteration [-]', 
ylabel='$J$ [-]',
xlim = [0,len(zlcFileName)]) 
ax2.locator_params(axis="y", nbins=4)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.locator_params(axis="x", nbins=4)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.legend()   

#  Save the figure
if saveFlag:
    # FileName: isothermComparison_<currentDateTime>_<GitCommitID_Current>_<modelFile>
    saveFileName = "isothermComparison_" + currentDT + "_" + gitCommitID + "_" + zlcFileName[-25:-12] + saveFileExtension
    savePath = os.path.join('..','simulationFigures',saveFileName)
    # Check if simulationFigures directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationFigures')):
        os.mkdir(os.path.join('..','simulationFigures'))
    plt.savefig (savePath)         
plt.show()

# Plot the kinetic constant as a function of mole fraction
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax1 = plt.subplot(1,1,1)        
for jj in range(len(temperature)):
    for kk in range(len(zlcFileName)):
        if kk == 0:
            labelText = str(temperature[jj])+' K'
        else:
            labelText = ''
        ax1.plot(y,kineticConstant_Macro[kk,:,jj],color='#'+colorForPlot[jj]) # Macropore resistance
        ax1.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.2,
                 label=labelText) # ALL

ax1.set(xlabel='$P$ [bar]', 
ylabel='$k$ [s$^\mathregular{-1}$]',
xlim = [0,1], ylim = [0, 1]) 
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)
ax1.legend()   