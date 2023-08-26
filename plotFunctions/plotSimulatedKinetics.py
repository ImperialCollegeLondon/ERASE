############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2023
# Python:   Python 3.7
# Authors:  Hassan Azzan
#
# Purpose:
# Simulated Responses for different isotherms and kinetic models
#
# Last modified:
# - 2023-08-24, HA: Initial creation
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
from computeEquilibriumLoading import computeEquilibriumLoading

plt.style.use('doubleColumn4Row.mplstyle') # Custom matplotlib style file

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Save flag for plot
saveFlag = False

# Save file extension
saveFileExtension = ".png"

# Plot color
colorsForPlot = ["#faa307","#d00000","#03071e"]*4
colorsForPlot = ['r','g','b']*4

markerForPlot = ["o"]*20

# Generate .npz file for python processing of the .mat file 
# Get the processed file names
    
# ZLC parameter model path
# Temperature (for each experiment)
# temperatureExp = [344.69, 325.39, 306.15]*4 # AC Experiments
temperatureExp = [298.15,308.15]

# Legend flag
useFlow = False

# Mass of sorbent and particle epsilon
adsorbentDensity = 1680 # CMS 3K
particleEpsilon = 0.61 # CMS 3K
massSorbent = 0.10 # CMS 3K
Rg = 8.314            

flagDesorption = True

# Volume of sorbent material [m3]
volSorbent = (massSorbent/1000)/adsorbentDensity

# Volume of gas chamber (dead volume) [m3]
volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

# Dead volume model
deadVolumeFile = [[['deadVolumeCharacteristics_20230321_1036_59cc206.npz', # 50A
                  'deadVolumeCharacteristics_20230321_1245_59cc206.npz']], # 51A
                  [['deadVolumeCharacteristics_20230321_1137_59cc206.npz', # 50B
                  'deadVolumeCharacteristics_20230321_1252_59cc206.npz']]] # 51B ZEOLITE Y CMS

# deadVolumeFile = 'deadVolumeCharacteristics_20220823_1542_e81a19e.npz' # DA LV
# Isotherm parameter reference
# parameterReference = load(parameterPath)["parameterReference"]
##############
downsampleData = True
##############

modelType = 'KineticSBMacro'
modelType = 'KineticOld'
modelType = 'Kinetic'

isothermModels = [[995*3/4,0.0001/np.exp(2.9e4/(Rg*298.15)),2.9e4,0,0,0],
                  [8.9*3/4,0.02/np.exp(4.6e4/(Rg*298.15)),4.6e4,0,0,0],
                  [4.18*3/4,0.5/np.exp(17.7e4/(Rg*298.15)),17.7e4,0,0,0]]

if modelType == 'KineticSBMacro':
    KineticModels =[[0.021/np.exp(-2e4/(Rg*298.15)),2e1,0,],
                [2*0.021/np.exp(-2e4/(Rg*298.15)),2e1,2*3,],
                [0,0,3,]]
elif modelType == 'KineticMacro':
    KineticModels =[[0.21/np.power(298.15,0.5),0],
                [0.21/np.power(298.15,0.5),0],
                [0.21/np.power(298.15,0.5),0]]
else:
    KineticModels =[[0.021,0],
                [2*0.021,2*3,],
                [0,3,]]
legendLines = ['$k_{micro}$ << $k_{macro}$',
               '$k_{micro}$ ~= $k_{macro}$',
               '$k_{micro}$ >> $k_{macro}$']
lineStyles = ['-','dashed']


y = np.linspace(0,1,4000)
isoLoading = np.zeros(len(y))
# Create the instance for the plots
fig = plt.figure
# ax1 = plt.subplot(1,3,1)        
# ax2 = plt.subplot(1,3,2)
# ax3 = plt.subplot(1,3,3)

for jj in range(len(isothermModels)):
    if modelType == 'KineticSBMacro':
        x = np.zeros(9)
        x[0:6] = isothermModels[jj]
    else:
        x = np.zeros(8)
        x[0:6] = isothermModels[jj]
    for nn in range(len(temperatureExp)):
        linestyle = lineStyles[nn]
        for kk in range(len(y)):
            isoLoading[kk] = computeEquilibriumLoading(isothermModel=isothermModels[jj],
                                  moleFrac = y[kk],
                                  temperature = temperatureExp[nn]) # [mol/kg]
        
        plt.subplot(4,3,jj+1) .plot(y,isoLoading,color='k'
                                    ,linestyle = linestyle
                                    ,label=str(temperatureExp[nn])+' K')
        plt.subplot(4,3,jj+1) .set(xlabel='$P$ [bar]', 
        ylabel='$q^*$ [mol kg$^\mathregular{-1}$]',
        xlim = [0,1], ylim = [0, 3]) 
        plt.subplot(4,3,jj+1) .locator_params(axis="x", nbins=4)
        plt.subplot(4,3,jj+1) .locator_params(axis="y", nbins=4)
        if jj+1 == 1:
            plt.subplot(4,3,jj+1) .legend()
        for ii in range(len(KineticModels)):
            # Parse out parameter values
            if modelType == 'KineticSBMacro':
                x[-3:] = KineticModels[ii]
                isothermModel = x[0:-3]
                rateConstant_1 = x[-3]
                rateConstant_2 = x[-2]
                rateConstant_3 = x[-1]     
            else:
                x[-2:] = KineticModels[ii]
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = x[-1]
                rateConstant_3 = 0     
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = x[-1]
                rateConstant_3 = 0
            rateConstantZLC = np.zeros(len(y))
            temperature = temperatureExp[nn]
            for mm in range(len(y)):
                pressureTotal = 1e5
                equilibriumLoading  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                        temperature=temperatureExp[nn],
                                                        moleFrac=y[mm],
                                                        isothermModel=isothermModel)*adsorbentDensity # [mol/m3]
                # Partial pressure of the gas
                partialPressure = y[mm]*pressureTotal
                # delta pressure to compute gradient
                delP = 10
                # Mole fraction (up)
                moleFractionUp = (partialPressure + delP)/pressureTotal
                # Compute the loading [mol/m3] @ moleFractionUp
                equilibriumLoadingUp  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                                temperature=temperatureExp[nn],
                                                                moleFrac=moleFractionUp,
                                                                isothermModel=isothermModel)*adsorbentDensity # [mol/m3]
            
                # Compute the gradient (delq*/dc)
                dqbydc = (equilibriumLoadingUp-equilibriumLoading)/(delP/(Rg*temperatureExp[nn])) # [-]
                dellogc = np.log(partialPressure+delP)-np.log((partialPressure))
                dlnqbydlnc = (np.log(equilibriumLoadingUp)-np.log(equilibriumLoading))/dellogc
                epsilonp = volGas/(volGas+volSorbent)
                
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
                    k1 = rateConstant_1/(1+(1/epsilonp)*dqbydc)*np.power(temperature,0.5)
                    k2 = rateConstant_2/(1+(1/epsilonp)*dqbydc)*np.power(temperature,1.5)/partialPressure
                    if k1<1e-9:
                        rateConstant = k2
                    # If pure (analogous) micropore
                    elif k2<1e-9:
                        rateConstant = k1
                    # If both resistances are present
                    else:
                        rateConstant = 1/(1/k1 + 1/k2)
                        
                elif modelType == 'KineticSB':
                    rateConstant = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))/dlnqbydlnc
                    if rateConstant<1e-8:
                        rateConstant = 1e-8
                elif modelType == 'KineticSBMacro':
                    k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))/dlnqbydlnc
                    # Rate constant 2 (analogous to macropore resistance)
                    k2 = rateConstant_3/(1+(1/epsilonp)*dqbydc)
                    
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
                rateConstantZLC[mm] = rateConstant
                    
                
            
            timeElapsedExp = np.linspace(0,2000,2000)
            timeInt = timeElapsedExp
            moleFracExp = np.linspace(1,1,2000)
            flowRateExp = np.linspace(10/60,10/60,2000)
            
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
                    
            if flagDesorption:
                initMoleFrac = [moleFracExp[0]]
                feedMoleFrac = [0]
            else:
                initMoleFrac = [1e-7]
                feedMoleFrac = [1]
            
            # initMoleFrac = [1e-6]
            # feedMoleFrac = [1]
            
            # Compute the combined zlc and dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = initMoleFrac, # Initial mole fraction assumed to be the first experimental point
                                                        flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                        feedMoleFrac = feedMoleFrac,
                                                        expFlag = True,
                                                        isothermModel = isothermModel,
                                                        rateConstant_1 = rateConstant_1,
                                                        rateConstant_2 = rateConstant_2,
                                                        rateConstant_3 = rateConstant_3,
                                                        deadVolumeFile = deadVolumeFileTemp,
                                                        volSorbent = volSorbent,
                                                        volGas = volGas,
                                                        temperature = temperatureExp[nn],
                                                        adsorbentDensity = adsorbentDensity,
                                                        modelType = modelType)
            
            
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFileTemp)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            flowInDV = np.zeros((len(resultMat[1,:])))          
            flowInDV[:] = np.mean(flowRateExp[-1:-2:-1]*1e-6)
            moleFracDV = deadVolumeWrapper(timeInt, flowInDV*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = initMoleFrac, feedMoleFrac = feedMoleFrac)
            
            internalLoading = resultMat[1,:]
            if flagDesorption:
                fractionalLoading = resultMat[1,:]/resultMat[1,0]
            else:
                fractionalLoading = resultMat[1,:]/resultMat[1,-1]
            # k - Pressure scale
            legendStr = legendLines[ii]
            
            if jj+4 == 4:
                if nn == 0:
                    plt.subplot(4,3,jj+4) .plot(y,rateConstantZLC,
                     color=colorsForPlot[ii],label=legendStr,alpha = 1, linestyle = linestyle) # Simulation response    
            else:
                plt.subplot(4,3,jj+4) .plot(y,rateConstantZLC,
                 color=colorsForPlot[ii],alpha = 1, linestyle = linestyle) # Simulation response  
                            
            
            if jj == 2:
                plt.subplot(4,3,jj+4) .set(xlabel='$P$ [bar]', 
                                           ylabel='$k$ [s$^\mathregular{-1}$]',
                                            xlim = [0,1], ylim = [0, 0.5]
                                           ) 
            else:
                plt.subplot(4,3,jj+4) .set(xlabel='$P$ [bar]', 
                                           ylabel='$k$ [s$^\mathregular{-1}$]',
                                            xlim = [0,1], ylim = [0, 0.05]
                                           ) 
            plt.subplot(4,3,jj+4) .locator_params(axis="x", nbins=4)
            plt.subplot(4,3,jj+4) .locator_params(axis="y", nbins=5)
            plt.subplot(4,3,4) .legend()
            
            # y - Linear scale
            legendStr = legendLines[ii]
            # plt.subplot(3,3,jj+4) .semilogy(timeElapsedExp,resultMat[0,:],
            plt.subplot(4,3,jj+7) .semilogy(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],label=legendStr,alpha = 1, linestyle = linestyle) # Simulation response    
            # if ii==len(fileName)-1:
            plt.subplot(4,3,jj+7) .plot(timeElapsedExp,moleFracDV,
                          color='#118ab2',alpha=0.2,
                          linestyle = '-') # Dead volume simulation response    
            
            if flagDesorption:
                plt.subplot(4,3,jj+7) .set(xlabel='$t$ [s]', 
                        ylabel='$y$ [-]', 
                        xlim = [0,2000], ylim =  [1e-6, 1])  
            else:
                plt.subplot(4,3,jj+7) .set(xlabel='$t$ [s]', 
                        ylabel='$y$ [-]', 
                        xlim = [0,200], ylim =  [1e-6, 1]) 
            plt.subplot(4,3,jj+7) .locator_params(axis="x", nbins=4)
    
            
            
            # fract loading
            plt.subplot(4,3,jj+10) .plot(timeElapsedExp,fractionalLoading,
                     color=colorsForPlot[ii],alpha = 1, linestyle = linestyle) # Simulation response    
     
            if flagDesorption:
                plt.subplot(4,3,jj+10) .set(xlabel='$t$ [s]', 
                        ylabel='$q^*$(t)/$q^*_0$ [-]',  
                        xlim = [0,2000], ylim =  [0, 1])   
            else:
                plt.subplot(4,3,jj+10) .set(xlabel='$t$ [s]', 
                        ylabel='$q^*$(t)/$q^*_0$ [-]',  
                        xlim = [0,200], ylim =  [0, 1])  
            plt.subplot(4,3,jj+10) .locator_params(axis="x", nbins=4)
        
plt.show()
