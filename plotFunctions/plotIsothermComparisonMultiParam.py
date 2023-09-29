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
colorForPlot = ["faa307","d00000","03071e","d00000","03071e"]
# colorForPlot = ["E5383B","6C757D"]

# Plot text
plotText = 'DSL'

# Universal gas constant
Rg = 8.314

# Total pressure
pressureTotal = np.array([1.e5]);

# Define temperature
# temperature = [ 363.15, 348.15, 333.15, 318.15, 303.15,]
temperature = [308.15, 328.15, 348.15]
# temperature = [ 283.15, 293.15, 303.15,  ] # ZIF8 BN
temperature = [ 288.15, 298.15, 308.15, ] # ZYH 

# CO2 molecular diffusivity
molDiffusivity = 1.6e-5 # m2/s

# Particle Tortuosity
tortuosity = 3

# Particle Radius
particleRadius = 2e-3

# AC Isotherm parameters
x_VOL = [4.65e-1, 1.02e-5 , 2.51e4, 6.51, 3.51e-7, 2.57e4] # (Hassan, QC)

# Lewatit Isotherm parameters
# x_VOL = [1.02, 2.18e-10,  5e4, 1.46, 1.92e-13, 8.44e4] # (Hassan, QC)

# 13X Isotherm parameters (L pellet)
# x_VOL = [2.50, 2.05e-7, 4.29e4, 4.32, 3.06e-7, 3.10e4] # (Hassan, QC)
# x_VOL = [3.83, 1.33e-08, 40.0e3, 2.57, 4.88e-06, 35.16e3] # (Hassan, QC - Bound delU to 40e3)

# BN Isotherm parameters
# x_VOL = [7.01, 2.32e-07, 2.49e4, 0, 0, 0] # (Hassan, QC)

# ZIF8 MT Isotherm parameters
# x_VOL = [20, 5.51151743793683e-07, 19272.6327577962,1.12005137776376] # (Hassan, QC)

# ZIF8 MCB20 Isotherm parameters
# x_VOL = [7.94, 3.42794467426045e-07, 21424.1416899692,0, 0, 0] # (Hassan, QC)
# 
# ZYH Isotherm parameters
x_VOL =  [4.3418e-01, 1.0555e-06 , 3.2322e+04, 6.6351e+00, 2.0803e-07, 2.6108e+04] # ZYH DSL 1 bara new
# x_VOL =  [20, 2.0760e-08, 2.7017e+04, 6.6713e-01] # ZYH SSS 1 bara
# 
# ZYNa Isotherm parameters
x_VOL =  [6.4975e+00, 3.5355e-07, 3.1108e+04, 9.0420e-01, 5.1101e-05, 2.3491e+04] # ZYNa DSL 1 bara new
# ZYTMA VOL
# x_VOL = [5.1394e+00, 2.7760e-07, 2.8886e+04, 2.6934e+00, 1.2966e-06, 2.9451e+04] # ZYTMA DSL 1 bara new

# ZYNa Isotherm parameters
# x_VOL =  [6.1805e+00, 3.5221e-07, 3.1113e+04, 8.6948e-01, 4.9984e-05, 2.3522e+04] # ZYNa DSL 1 bara

# CMS 3K Isotherm parameters
x_VOL =  [9.7667e-01, 1.0749e-05, 2.4312e+04, 3.4760e+00, 2.2254e-07, 2.7095e+04]  # CMS3K DSL 1 bara new

# ZIF8 MCB30 Isotherm parameters
# x_VOL = [10,4.41e-8,2.57e4,0, 0, 0] # (Hassan, QC)

# 13X Isotherm parameters (L pellet)
x_VOL = [2.80321673e+00, 2.35760556e-08, 4.54444851e+04, 3.50928608e+00, 3.73556076e-10, 4.72948894e+04,]
# ZLC Parameter estimates
# New kinetic model
# Both k1 and k2 present


# fileParameter = 'zlcParameters_ZYH_20230914_0359_b571c46.npz' # ZYH ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_ZYNa_20230914_1950_b571c46.npz' # ZYNa ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_ZYTMA_20230915_1651_b571c46.npz' # ZYTMA ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_Zeolite13X_20230924_1314_b571c46.npz' # 13X ALL FLOW SBMACRO
# fileParameter = 'zlcParameters_CMS3K_20230919_1800_b571c46.npz' # CMS He ALL FLOW SBMACRO high comp
# fileParameter = 'zlcParameters_CMS3KAr_20230920_0458_b571c46.npz' # CMS Ar ALL FLOW SBMACRO high comp
# fileParameter = 'zlcParameters_ActivatedCarbon_20230921_0603_b571c46.npz' # AC ALL FLOW SBMACRO high comp


# Activated Carbon Experiments
zlcFileName = ['zlcParameters_20210822_0926_c8173b1.npz',
                'zlcParameters_20210822_1733_c8173b1.npz',
                'zlcParameters_20210823_0133_c8173b1.npz',
                'zlcParameters_20210823_1007_c8173b1.npz',
                'zlcParameters_20210823_1810_c8173b1.npz']

zlcFileName = ['zlcParameters_ZYH_20230914_0359_b571c46.npz',] 
# zlcFileName = ['zlcParameters_ZYTMA_20230915_1651_b571c46.npz',] 
# zlcFileName = ['zlcParameters_ZYNa_20230914_1950_b571c46.npz',] 
# zlcFileName = ['zlcParameters_Zeolite13X_20230924_1314_b571c46.npz',] 
# zlcFileName = ['zlcParameters_CMS3K_20230919_1800_b571c46.npz',] 
# zlcFileName = ['zlcParameters_CMS3KAr_20230920_0458_b571c46.npz',] 
# zlcFileName = ['zlcParameters_ActivatedCarbon_20230921_0603_b571c46.npz',] 

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

# # Activated Carbon Simulations (Main)
# zlcFileName = ['zlcParameters_20220428_1624_e81a19e.npz',
#                'zlcParameters_20220430_0508_e81a19e.npz',
#                'zlcParameters_20220501_1942_e81a19e.npz',
#                'zlcParameters_20220429_0445_e81a19e.npz',
#                'zlcParameters_20220430_1806_e81a19e.npz',
#                'zlcParameters_20220502_0753_e81a19e.npz',
#                'zlcParameters_20220429_1657_e81a19e.npz',
#                'zlcParameters_20220501_0653_e81a19e.npz',
#                'zlcParameters_20220502_2004_e81a19e.npz',]

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

# ZIF8 MT Experiments
# zlcFileName = ['zlcParameters_20221130_1509_7e5a5aa.npz',
#                'zlcParameters_20221130_2332_7e5a5aa.npz',]
# ZIF8 MT Experiments (all comp)
# Downsampling Conc: False
# Downsampling Exp Points: False
# zlcFileName = ['zlcParameters_20221201_1102_7e5a5aa.npz',
#                'zlcParameters_20221201_1924_7e5a5aa.npz',
#                'zlcParameters_20221202_0357_7e5a5aa.npz',
#                'zlcParameters_20221202_1214_7e5a5aa.npz',
#                'zlcParameters_20221202_2021_7e5a5aa.npz',]

# ZIF8 MT Experiments (all comp)
# Downsampling Conc: True
# Downsampling Exp Points: False
# zlcFileName = ['zlcParameters_20221209_1552_7e5a5aa.npz',
#                'zlcParameters_20221210_0111_7e5a5aa.npz',
#                'zlcParameters_20221210_0951_7e5a5aa.npz',
#                'zlcParameters_20221210_1840_7e5a5aa.npz',
#                'zlcParameters_20221211_0338_7e5a5aa.npz',]

# ZIF8 MT Experiments (high comp)
# Downsampling Conc: True
# Downsampling Exp Points: False
# zlcFileName = ['zlcParameters_20221207_1453_7e5a5aa.npz',
#                'zlcParameters_20221208_1241_7e5a5aa.npz',]

# ZIF8 MT Experiments (high comp)
# Downsampling Conc: False
# Downsampling Exp Points: False
# zlcFileName = ['zlcParameters_20221209_1552_7e5a5aa.npz',
#                'zlcParameters_20221210_0111_7e5a5aa.npz',
#                'zlcParameters_20221210_0951_7e5a5aa.npz',
#                'zlcParameters_20221210_1840_7e5a5aa.npz',
#                'zlcParameters_20221211_0338_7e5a5aa.npz',]
# zlcFileName = ['zlcParameters_20221209_1552_7e5a5aa.npz',]


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

# Zeolite YH experiments (SSL)
# zlcFileName = ['zlcParameters_20230311_1325_7e5a5aa.npz',
#                 'zlcParameters_20230311_2126_7e5a5aa.npz',
#                 'zlcParameters_20230312_0523_7e5a5aa.npz',
#                 'zlcParameters_20230312_1319_7e5a5aa.npz',
#                 'zlcParameters_20230312_2111_7e5a5aa.npz']

# # Zeolite YH experiments (DSL)
# zlcFileName = ['zlcParameters_20230313_1858_7e5a5aa.npz',
#                 'zlcParameters_20230314_0031_7e5a5aa.npz',]

# # Zeolite YH experiments (SSS) ALL
# zlcFileName = ['zlcParameters_20230314_1720_59cc206.npz',
#                 'zlcParameters_20230315_0130_59cc206.npz',
#                 'zlcParameters_20230315_0921_59cc206.npz',
#                 'zlcParameters_20230315_1700_59cc206.npz',
#                 'zlcParameters_20230316_0108_59cc206.npz']

# # Zeolite YH experiments (SSS) ALL
# zlcFileName = ['zlcParameters_20230315_0921_59cc206.npz',]

# # Zeolite YH experiments (SSS) HIGH COMP
# zlcFileName = ['zlcParameters_20230316_1000_59cc206.npz',
#                 'zlcParameters_20230316_1523_59cc206.npz',
#                 'zlcParameters_20230316_2045_59cc206.npz',
#                 'zlcParameters_20230317_0209_59cc206.npz',
#                 'zlcParameters_20230317_0733_59cc206.npz',]

# # Zeolite YH experiments (SSS) HIGH COMP
# zlcFileName = [ 'zlcParameters_20230317_0733_59cc206.npz',]

# # Zeolite YH experiments (SSS) ALL COMP KINETICS ONLY
# zlcFileName = ['zlcParameters_20230317_1758_59cc206.npz',
#                 'zlcParameters_20230318_0233_59cc206.npz',
#                 'zlcParameters_20230318_1112_59cc206.npz',
#                 'zlcParameters_20230318_1950_59cc206.npz',
#                 'zlcParameters_20230319_0426_59cc206.npz',]

# Zeolite YH experiments (SSS) HIGH COMP KINETICS ONLY
# zlcFileName = ['zlcParameters_20230324_0100_59cc206.npz',]

# Zeolite YNa experiments (SSS) HIGH COMP KINETICS ONLY
# zlcFileName = ['zlcParameters_20230330_1528_59cc206.npz',]

# Zeolite TMA experiments (SSS) HIGH COMP ALL
# zlcFileName =['zlcParameters_20230707_0921_b571c46.npz',
#                 'zlcParameters_20230707_1637_b571c46.npz',
#                 'zlcParameters_20230707_2359_b571c46.npz',
#                 'zlcParameters_20230708_0721_b571c46.npz',
#                 'zlcParameters_20230708_1450_b571c46.npz',]

y = np.linspace(0,1.,1000)
k1valsFull = np.zeros([len(zlcFileName),len(y),len(temperature)])
k2valsFull = np.zeros([len(zlcFileName),len(y),len(temperature)])
# Initialize isotherms 
parameterPath = os.path.join('..','simulationResults',zlcFileName[0])
temperatureExp = load(parameterPath)["temperature"]
temperature = np.unique(temperatureExp)
isoLoading_VOL = np.zeros([len(y),len(temperature)])
isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
kineticConstant_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
kineticConstant_Macro = np.zeros([len(zlcFileName),len(y),len(temperature)])
objectiveFunction = np.zeros([len(zlcFileName)])
Tvals = np.linspace(np.min(temperature),np.max(temperature),len(y))

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
    # parameterReference = [20, 1e-5, 40e3, 1000, 1000]
    modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
    objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
    modelNonDim = modelOutputTemp[()]["variable"] 
    modelType = load(parameterPath)["modelType"]
    paramIso = load(parameterPath)["paramIso"]
 #    modelNonDim = [0.19988476, 0.50978288 ,0.64060534, 0.42589773, 0.00479913 ,0.87887948,
 # 0.59783281, 0.02982962]
    # modelNonDim = [0.62367463, 0.03599396, 0.64567045, 0.28883399, 0.24600739, 0.03936779]
    # Multiply the paremeters by the reference values
    kineticConstants = np.multiply(modelNonDim,parameterReference)
    # x_ZLC = np.zeros(8)
    if modelType == 'KineticSBMacro':
        x_ZLC = np.zeros(9)
    else:
        x_ZLC = np.zeros(8)
    # x_ZLC = np.zeros(6)
    # x_ZLC[-2:] = np.multiply(modelNonDim,parameterReference)
    # x_ZLC[0:3] = [2.99999956e+01, 9.62292726e-08, 2.15988572e+04,]# MT
    # x_ZLC[3:5] = [1.70e-01, 9.83e+02] # MT
    # x_ZLC[0:3] = [7.94, 3.42794467426045e-07, 21424.1416899692,]# MCB20
    # x_ZLC[3:5] = [0.00012402*1000, 0.00053755*1000] # MCB20 downsample fals
    # x_ZLC[0:3] = [10,4.41e-8,2.57e4] # MCB30
    # x_ZLC[3:5] = [0.0011*1000, 0.00528*1000] # MCB30
    # x_ZLC[3:5] = [0.00097998*1000, 0.0053946*1000] # MCB30
    
    
    # x_ZLC[0:4] = [20, 5.51151743793683e-07, 19272.6327577962,1.12005137776376]# MT SSS 
    # x_ZLC[-2:] = [1.70e-01, 9.83e+02] # MT
    # x_ZLC[-2:] = [0.2445, 983] # MT HIGH FLOW
    # x_ZLC[-2:] = [0.29, 983] # MT ALL FLOW 
    
    # x_ZLC[0:4] = [9.6369, 2.74e-7, 2.14e+04,0.9893]# MCB20 SSS
    # x_ZLC[-2:] = [9.89720284e-02, 9.56339260e-01] # MCB20 SSS LOW FLOW
    # x_ZLC[-2:] = [0.22544, 1.2982] # MCB20 SSS ALL FLOW
    # x_ZLC[-2:] = [601, 0.645] # MCB20 SSS ALL FLOW HIGH COMP ONLY 


    # x_ZLC[0:4] = [6.51504325937662, 1.11248296208017e-07, 24942.1235836143,1.06047499753496]# MCB30 SSS
    # x_ZLC[-2:] = [1.09207, 2.9566] # MCB30 SSS ALL FLOW
    # x_ZLC[0:6] = [4.9140e-01, 1.0513e-06 , 3.2337e+04, 7.5119e+00, 2.0834e-07, 2.6107e+04]  # ZYH 
    # x_ZLC[-2:] = [1.10299208, 42.05108328] # ZYH

    # x_ZLC[0:6] = [6.1805e+00, 3.5221e-07, 3.1113e+04, 8.6948e-01, 4.9984e-05, 2.3522e+04]  # ZYNa
    # x_ZLC[-2:] = [4.53948237e-01, 5.83281350e+01] # ZYNa
    # x_ZLC[0:6] =  [4.3363e-01, 9.9690e-07 , 3.2471e+04, 6.6293e+00, 2.0871e-07, 2.6103e+04]# MCB30 SSS
    
    # x_ZLC[0:6] = [4.3418e-01, 1.0555e-06 , 3.2322e+04, 6.6351e+00, 2.0803e-07, 2.6108e+04] # ZYH DSL 1 bara new
    # x_ZLC[0:6] = [6.4975e+00, 3.5355e-07, 3.1108e+04, 9.0420e-01, 5.1101e-05, 2.3491e+04] # ZYNa DSL 1 bara new
    # x_ZLC[0:6] = [5.1394e+00, 2.7760e-07, 2.8886e+04, 2.6934e+00, 1.2966e-06, 2.9451e+04] # ZYTMA DSL 1 bara new

    # x_ZLC[-3:] = [1.41299542e+02*1e3, 3.62091593e-02*1e3, 9.18446265e-01*1e3] # SBmacro run 1 best ZYH
    # x_ZLC[-3:] = [8.50392278e+03, 3.19273655e+01,  9.99707560e+02] # SBmacro run 1 best ZYNa
    
    # x_ZLC[-3:] = [1.48859751e+05, 3.63321115e+01, 8.06733749e+02] # SBmacro run 1 best ZYH
    # x_ZLC[-3:] = [8503.92277915,   31.92736554,  999.70755954] # SBmacro run 1 best ZYNa
    # x_ZLC[-3:] = [1.49471854e+05, 3.83405715e+01, 9.95895797e+02] # SBmacro run 1 best ZYTMA
    
    # x_ZLC[-2:] =  [0.00012277*1000, 0.02934571*1000] # ZYTMA high comp OPT
    # x_ZLC[-2:]= [6.70694855e-02, 5.49592656e01] # ZYTMA high comp new kin
    
    # x_ZLC[0:6] = [9.7667e-01, 1.0749e-05, 2.4312e+04, 3.4760e+00, 2.2254e-07, 2.7095e+04]  # CMS3K DSL 1 bara new
    # x_ZLC[-3:] = [0.17716856*1e3, 0.02432568*1e3, 0] # SB run 1 best Helium
    # x_ZLC[-3:] =  [0.2125*1e3, 0.0246389*1e3, 0] # SB run 1 best Ar
    # x_ZLC[-3:] =  [2.62506226e+03, 3.10025426e+01,0] # CMS He
    # x_ZLC[-3:] =  [0.25*1e3, 0.0250343*1e3,0] # CMS Ar
    # x_ZLC = [1.99884760e+00, 5.09782880e-06, 2.56242136e+04, 4.25897730e+00,
       # 4.79913000e-08, 3.51551792e+04, 5.97832810e+02, 2.98296200e+01]
    # kineticConstants[-1] = 0
    print(x_ZLC)

    adsorbentDensity = load(parameterPath, allow_pickle=True)["adsorbentDensity"]
    particleEpsilon = load(parameterPath)["particleEpsilon"]
    epsilonp = particleEpsilon
    # Print names of files used for the parameter estimation (sanity check)
    fileNameList = load(parameterPath, allow_pickle=True)["fileName"]
    print(fileNameList)
    
    # Parse out the isotherm parameter
    # Parse out parameter values
    if modelType == 'KineticSBMacro':
        isothermModel = paramIso[0:-3]
        rateConstant_1 = kineticConstants[-3]
        rateConstant_2 = kineticConstants[-2]
        rateConstant_3 = kineticConstants[-1]  
    elif modelType == 'KineticMacro':
        isothermModel = paramIso[0:-3]
        rateConstant_1 = kineticConstants[-3]
        rateConstant_2 = kineticConstants[-2]
        rateConstant_3 = kineticConstants[-1]  
    else:
        isothermModel = paramIso[0:-2]
        rateConstant_1 = kineticConstants[-2]
        rateConstant_2 = kineticConstants[-1]
        rateConstant_3 = 0     

    # rateConstant_2 = 0
    

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
            dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[kk,ii,jj])/(delP/(Rg*temperature[jj])) # [-]
            dellogc = np.log(partialPressure+delP)-np.log((partialPressure))
            dlnqbydlnc = (np.log(equilibriumLoadingUp)-np.log(isoLoading_ZLC[kk,ii,jj]))/dellogc
            
            # Overall rate constant
            # The following conditions are done for purely numerical reasons
            # If pure (analogous) macropore
            if modelType == 'KineticOld':
            # Rate constant 1 (analogous to micropore resistance)
                k1 = rateConstant_1
            
                # Rate constant 2 (analogous to macropore resistance)
                k2 = rateConstant_2/dqbydc
                
                # Overall rate constant
                # The following conditions are done for purely numerical reasons
                # If pure (analogous) macropore
                if k1<1e-9:
                    rateConstant = k2
                # If pure (analogous) micropore
                elif k2<1e-9:
                    rateConstant = k1
                # If both resistances are present
                else:
                    rateConstant = 1/(1/k1 + 1/k2)
                    
            if modelType == 'Kinetic':
            # Rate constant 1 (analogous to micropore resistance)
                k1 = rateConstant_1/dlnqbydlnc
            
                # Rate constant 2 (analogous to macropore resistance)
                k2 = rateConstant_2/(1+(1/epsilonp)*dqbydc)
                
                # Overall rate constant
                # The following conditions are done for purely numerical reasons
                # If pure (analogous) macropore
                if k1<1e-9:
                    rateConstant = k2
                # If pure (analogous) micropore
                elif k2<1e-9:
                    rateConstant = k1
                # If both resistances are present
                else:
                    rateConstant = 1/(1/k1 + 1/k2)
                    
            elif modelType == 'KineticMacro':
                k1 = rateConstant_1/(1+(1/epsilonp)*dqbydc)*np.power(temperature[jj],0.5)
                k2 = rateConstant_2/(1+(1/epsilonp)*dqbydc)*np.power(temperature[jj],1.5)/partialPressure
                if k1<1e-9:
                    rateConstant = k2
                # If pure (analogous) micropore
                elif k2<1e-9:
                    rateConstant = k1
                # If both resistances are present
                else:
                    rateConstant = 1/(1/k1 + 1/k2)
                    
            elif modelType == 'KineticSB':
                rateConstant = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature[jj]))/dlnqbydlnc
                k1 = rateConstant
                k2 = 0
                if rateConstant<1e-8:
                    rateConstant = 1e-8
        
            elif modelType == 'KineticSBMacro':
                k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature[jj]))/dlnqbydlnc
                # Rate constant 2 (analogous to macropore resistance)
                k2 = rateConstant_3*np.power(temperature[jj],0.5)/(1+(1/epsilonp)*dqbydc)
                # k2 = rateConstant_3/(1+(1/epsilonp)*dqbydc)
                
                # Overall rate constant
                # The following conditions are done for purely numerical reasons
                # If pure (analogous) macropore
                if k1<1e-9:
                    rateConstant = k2
                # If pure (analogous) micropore
                elif k2<1e-9:
                    rateConstant = k1
                # If both resistances are present
                else:
                    rateConstant = 1/(1/k1 + 1/k2)
            
                if rateConstant<1e-8:
                    rateConstant = 1e-8   
            k1valsFull[kk,ii,jj] =  k1
            k2valsFull[kk,ii,jj] =  k2
            
            k1vals =  rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*Tvals))
            k2vals =  rateConstant_3*np.power(Tvals,0.5)
            
            # Rate constant (overall)
            kineticConstant_ZLC[kk,ii,jj] = rateConstant
        
            # Macropore resistance from QC data
            # Compute dqbydc for QC isotherm
            equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature[jj],
                                                moleFrac=moleFractionUp,
                                                isothermModel=x_VOL) # [mol/kg]
            dqbydc_True = (equilibriumLoadingUp-isoLoading_VOL[ii,jj])*adsorbentDensity/(delP/(Rg*temperature[jj])) # [-]

            # Macropore resistance
            kineticConstant_Macro[kk,ii,jj] = (15*particleEpsilon*molDiffusivity
                                                /(tortuosity*(particleRadius)**2)/dqbydc_True)
            
# Plot the isotherms    
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax1 = plt.subplot(1,1,1)        
for jj in range(len(temperature)):
    # ax1.plot(y,isoLoading_VOL[:,jj],color='#'+colorForPlot[jj],label=str(temperature[jj])+' K') # Ronny's isotherm
    for kk in range(len(zlcFileName)):
        ax1.plot(y,isoLoading_ZLC[kk,:,jj],color='#'+colorForPlot[jj],alpha=1) # ALL

ax1.set(xlabel='$P$ [bar]', 
ylabel='$q^*$ [mol kg$^\mathregular{-1}$]',
xlim = [0,1], ylim = [0, 6]) 
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=4)
ax1.legend()   

# # Plot the objective function
# fig = plt.figure
# ax2 = plt.subplot(1,2,2)       
# for kk in range(len(zlcFileName)):
#     ax2.scatter(kk+1,objectiveFunction[kk]) # ALL

# ax2.set(xlabel='Iteration [-]', 
# ylabel='$J$ [-]',
# xlim = [0,len(zlcFileName)]) 
# ax2.locator_params(axis="y", nbins=4)
# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.locator_params(axis="x", nbins=4)
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.legend()   

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
        ax1.plot(y,kineticConstant_Macro[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.0) # Macropore resistance
        ax1.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.5,
                 label=labelText) # ALL

ax1.set(xlabel='$P$ [bar]', 
ylabel='$k$ [s$^\mathregular{-1}$]',
xlim = [0,1], ylim = [0, 1]) 
ax1.locator_params(axis="x", nbins=4)
ax1.locator_params(axis="y", nbins=5)
ax1.legend()   
plt.show()


# Plot the Arrhenius plot for k1
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax1 = plt.subplot(1,1,1)        

ax1.plot(1/Tvals,np.log(k1vals),color='#'+colorForPlot[0],alpha=1) # Macropore resistance

ax1.set(xlabel='$1/T$ [1/K]', 
ylabel='$ln k_1$ [-]',
xlim = [0.0025,0.004], ylim = [-7, 0]) 
ax1.locator_params(axis="x", nbins=4)
# ax1.locator_params(axis="y", nbins=5)
# ax1.legend()
plt.show()

# Plot the Arrhenius plot for k1
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax1 = plt.subplot(1,1,1)        

ax1.plot(np.power(Tvals,0.5),k2vals,color='#'+colorForPlot[0],alpha=1) # Macropore resistance

ax1.set(xlabel='$T^{0.5}$', 
ylabel='$k_2$ [-]',
xlim = [15,20], ylim = [0, 1.1*np.max(k2vals)]) 
ax1.locator_params(axis="x", nbins=4)
# ax1.locator_params(axis="y", nbins=5)
# ax1.legend()
plt.show()

# Plot the ratio of kinetic constant as a function of mole fraction
plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
fig = plt.figure
ax1 = plt.subplot(1,1,1)        
for jj in range(len(temperature)):
    for kk in range(len(zlcFileName)):
        if kk == 0:
            labelText = str(temperature[jj])+' K'
        else:
            labelText = ''
        ax1.semilogy(y,k1valsFull[kk,:,jj]/k2valsFull[kk,:,jj],color='#'+colorForPlot[jj],alpha=1, lineStyle = '--') #
        ax1.semilogy(y,k1valsFull[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.5) # ALL
        ax1.semilogy(y,k2valsFull[kk,:,jj],color='#'+colorForPlot[jj],alpha=0.5,
                 label=labelText) # ALL

ax1.set(xlabel='$P$ [bar]', 
ylabel='$k_1/k_2$ [-]',
xlim = [0,1],) 
ax1.locator_params(axis="x", nbins=4)
# ax1.locator_params(axis="y", nbins=5)
ax1.legend()   
plt.show()