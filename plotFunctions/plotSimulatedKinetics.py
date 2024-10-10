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
import pdb

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
temperatureExp = [298.15]

# Legend flag
useFlow = False

# Mass of sorbent and particle epsilon
adsorbentDensity = 1750 #
particleEpsilon = 0.5 # 
massSorbent = 0.10 # 
Rg = 8.314            
Fin = 5/60
flagDesorption = True
flagDesorption = False

if flagDesorption:
    timeExp = 8000
else:
    timeExp = 8000
adsFraction = [0.05]
desFraction = 0.1
desFraction = 0.1

moleFracThres = 1e-8
# Volume of sorbent material [m3]
volSorbent = (massSorbent/1000)/adsorbentDensity

# Volume of gas chamber (dead volume) [m3]
volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

# Dead volume model
deadVolumeFile = [[['deadVolumeCharacteristics_20231122_1743_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1757_b571c46.npz']], 
                  [['deadVolumeCharacteristics_20231122_1750_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1804_b571c46.npz']]] 

# deadVolumeFile = 'deadVolumeCharacteristics_20220823_1542_e81a19e.npz' # DA LV
# Isotherm parameter reference
# parameterReference = load(parameterPath)["parameterReference"]
##############
downsampleData = True
##############

# modelType = 'KineticMacro'
# modelType = 'KineticOld'
modelType = 'KineticOld'
# modelType = 'Diffusion1Ttau'

isothermModels = [[995*6/4,0.0001/np.exp(2.9e4/(Rg*298.15)),2.9e4,0,0,0],
                  [6.5*6/4,0.04/np.exp(4.6e4/(Rg*298.15)),4.6e4,0,0,0],
                  [4.18*6/4,0.5/np.exp(17.7e4/(Rg*298.15)),17.7e4,0,0,0]]

# isothermModels = [[995*6/4,0.0001/np.exp(2.9e4/(Rg*298.15)),2.9e4,0,0,0],
#                   [6.5*6/4,0.04/np.exp(4.6e4/(Rg*298.15)),4.6e4,0,0,0],
#                   [6,500/np.exp(4.6e4/(Rg*298.15)),4.6e4,0,0,0]]

# [[6],[80/np.exp(4.6e4/(Rg*298.15))],[-4.6e4]]
# isothermModels = [[3.83, 1.33e-08, 40.0e3, 2.57, 4.88e-06, 35.16e3],
#                   [3.75,5.3767e-6,1.5472e4,0,0,0],
#                   [4.18*6/4,0.5/np.exp(17.7e4/(Rg*298.15)),17.7e4,0,0,0]]
if modelType == 'KineticSBMacro':
    KineticModels =[[0.04/np.exp(-2e4/(Rg*298.15)),2e1,0,],
                [0.04/np.exp(-2e4/(Rg*298.15)),2e1,41/(288**0.5),],
                [0.04/np.exp(-2e4/(Rg*298.15)),20,41/(288**0.5),]]
    # KineticModels =[[99,0,6/(288**0.5),],
    #             [99,0,1/(288**0.5),],
    #             [99,0,1.4/(288**0.5),]]
    # KineticModels =[[0.01,0,0,],
    #             [0.1,0,0,],
    #             [1,0,0,]]
elif modelType == 'KineticMacro':
    KineticModels =[[0.21/np.power(298.15,0.5),0],
                [0.21/np.power(298.15,0.5),0],
                [0.21/np.power(298.15,0.5),0]]
elif modelType == 'Diffusion1Ttau':
    KineticModels =[[100,2],
                [100,5,]]
else:
    KineticModels =[[10,0],
                [10,0,],
                [10,0,],]
legendLines = ['$k_{micro}$ << $k_{macro}$',
               '$k_{micro}$ ~= $k_{macro}$',
               '$k_{micro}$ >> $k_{macro}$']
legendLines = ['$k = 1$ s$^\mathregular{-1}$',
                '$k = 10$ s$^\mathregular{-1}$',
                '$k = 30$ s$^\mathregular{-1}$',]

# legendLines = ['$τ = 2$',
#                 '$τ = 5$',]
lineStyles = ['-','dashed']

MassBalanceALL = []

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
        ylabel='$n^*$ [mol kg$^\mathregular{-1}$]',
        xlim = [0,1], ylim = [0, 7]) 
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
            elif modelType == 'Diffusion1Ttau':
                x[-2:] = KineticModels[ii]
                isothermModel = x[0:-2]
                rateConstant_1 = x[-2]
                rateConstant_2 = 0
                rateConstant_3 = x[-1]    
            else:
                x[-2:] = KineticModels[ii]
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
                
                rpore = 107e-9
                Dpvals = [2.35952892668521e-05,	2.42488804831046e-05	,2.48936504671912e-05]
                Dpvals = [4.655e-5,	4.778e-05	,4.899e-05]
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
                    rateConstantZLC[mm] = rateConstant
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
                    rateConstantZLC[mm] = rateConstant
                        
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
                    rateConstantZLC[mm] = rateConstant
                        
                elif modelType == 'KineticSB':
                    rateConstant = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))/dlnqbydlnc
                    if rateConstant<1e-8:
                        rateConstant = 1e-8
                    rateConstantZLC[mm] = rateConstant
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
                    
                
            if jj == 0:
                timeElapsedExp = np.linspace(0,timeExp,4000)
            else:
                timeElapsedExp = np.linspace(0,timeExp,4000)
            timeInt = timeElapsedExp
            moleFracExp = np.linspace(desFraction,desFraction,4000)
            flowRateExp = np.linspace(Fin,Fin,4000)
            flowRateExp = np.linspace(Fin,Fin,4000)
            
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
                # initMoleFrac = [0.1]
                # feedMoleFrac = [0.05]
                feedMoleFrac = [0]
            else:
                initMoleFrac = [1e-7]
                feedMoleFrac = adsFraction
            
            # initMoleFrac = [1e-6]
            # feedMoleFrac = [1]
            
            # _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
            #                                             initMoleFrac = initMoleFrac, # Initial mole fraction assumed to be the first experimental point
            #                                             flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
            #                                             feedMoleFrac = feedMoleFrac,
            #                                             expFlag = True,
            #                                             isothermModel = isothermModel,
            #                                             rateConstant_1 = rateConstant_1,
            #                                             rateConstant_2 = rateConstant_2,
            #                                             rateConstant_3 = rateConstant_3,
            #                                             rpore = rpore,
            #                                             Dpvals = Dpvals,
            #                                             deadVolumeFile = deadVolumeFileTemp,
            #                                             volSorbent = volSorbent,
            #                                             volGas = volGas,
            #                                             temperature = temperatureExp[nn],
            #                                             adsorbentDensity = adsorbentDensity,
            #                                             modelType = 'DV')
            
            # flowInDV = resultMat[3,:]    
            # pdb.set_trace()
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
                                                        rpore = rpore,
                                                        Dpvals = Dpvals,
                                                        deadVolumeFile = deadVolumeFileTemp,
                                                        volSorbent = volSorbent,
                                                        volGas = volGas,
                                                        temperature = temperatureExp[nn],
                                                        adsorbentDensity = adsorbentDensity,
                                                        modelType = modelType)
            
            moleFracSim[moleFracSim<moleFracThres] = moleFracThres
            flowRateSim =  resultMat[3,:]
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFileTemp)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            flowInDV = np.zeros((len(resultMat[1,:])))          
            flowInDV[:] = np.mean(flowRateExp[-1:-2:-1]*1e-6)
            moleFracDV = deadVolumeWrapper(timeInt, flowRateSim*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = initMoleFrac, feedMoleFrac = feedMoleFrac)
            FeedFlow = Fin/(1e6)
            internalLoading = resultMat[1,:]
            if flagDesorption:
                fractionalLoading = resultMat[1,:]/resultMat[1,0]
            else:
                fractionalLoading = resultMat[1,:]/resultMat[1,-1]
            # y-t log scale
            legendStr = legendLines[ii]
            
            if flagDesorption:
                if jj+4 == 4:
                    if nn == 0:
                        plt.subplot(4,3,jj+4) .plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],label=legendStr,alpha = 1, linestyle = linestyle) # Simulation response 
                    else:
                        plt.subplot(4,3,jj+4) .plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],alpha = 1, linestyle = linestyle) # Simulation response 
                    plt.subplot(4,3,jj+4) .plot(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracDV,
                    color='#118ab2',alpha=0.2,
                    linestyle = '-') # Dead volume simulation response    
                    plt.subplot(4,3,jj+4) .set(xlabel='$t$ [s]', 
                        ylabel='$y$ [-]', 
                        # xlim = [0,2000], ylim =  [1e-6, 1])  
                        xlim = [0,1000], ylim =  [1e-2, 1])  
                else:
                    plt.subplot(4,3,jj+4) .plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],alpha = 1, linestyle = linestyle) # Simulation response 
                    # if ii==len(fileName)-1:
                    plt.subplot(4,3,jj+4) .plot(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracDV,
                        color='#118ab2',alpha=0.2,
                        linestyle = '-') # Dead volume simulation response    
                    plt.subplot(4,3,jj+4) .set(xlabel='$t$ [s]', 
                            ylabel='$y$ [-]', 
                            # xlim = [0,2000], ylim =  [1e-6, 1])  
                            xlim = [0,1000], ylim =  [1e-3, 1])  
            else:
                plt.subplot(4,3,jj+4) .plot(timeElapsedExp,moleFracSim,
                     color=colorsForPlot[ii],label=legendStr,alpha = 1, linestyle = linestyle) # Simulation response    
                    # if ii==len(fileName)-1:
            plt.subplot(4,3,jj+4) .plot(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracDV,
                color='#118ab2',alpha=0.2,
                linestyle = '-') # Dead volume simulation response    
            plt.subplot(4,3,jj+4) .set(xlabel='$t$ [s]', 
                    ylabel='$y$ [-]', ylim =  [0, 1]) 
            plt.subplot(4,3,jj+4) .locator_params(axis="x", nbins=4)
            plt.subplot(4,3,4) .legend()
            
            # Qy/(Qy)in - t log scale
            legendStr = legendLines[ii]
            
            # plt.subplot(3,3,jj+4) .semilogy(timeElapsedExp,resultMat[0,:],
            if flagDesorption:
                if jj+7 == 7:
                    if nn == 0:
                        plt.subplot(4,3,jj+7) .semilogy(timeElapsedExp,moleFracSim*flowRateSim/(initMoleFrac[0]*flowRateSim[0]),
                            color=colorsForPlot[ii],label=legendStr,alpha = 1, linestyle = linestyle) # Simulation response 
                    else:
                        plt.subplot(4,3,jj+7) .semilogy(timeElapsedExp,moleFracSim*flowRateSim/(initMoleFrac[0]*flowRateSim[0]),
                     color=colorsForPlot[ii],alpha = 1, linestyle = linestyle) # Simulation response 
                    plt.subplot(4,3,jj+7) .semilogy(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracDV*flowInDV/(initMoleFrac[0]*flowRateSim[0]),
                    color='#118ab2',alpha=0.2,
                    linestyle = '-') # Dead volume simulation response    
                    plt.subplot(4,3,jj+7) .set(xlabel='$t$ [s]', 
                            ylabel='$Q(t)y(t)/QinYin$ [-]') 
                    plt.fill_between(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracSim*flowRateSim/(initMoleFrac[0]*flowRateSim[0]),
                                     moleFracDV*flowInDV/(initMoleFrac[0]*flowRateSim[0]),alpha = 0.1,color = 'red')
                    plt.subplot(4,3,jj+7) .set(xlabel='$t$ [s]', 
                            ylabel='$Q(t)y(t)/QinYin$ [-]', ylim =  [1e-5, 1]) 
                else:
                    plt.subplot(4,3,jj+7) .semilogy(timeElapsedExp,moleFracSim*flowRateSim/(initMoleFrac[0]*flowRateSim[0]),
                     color=colorsForPlot[ii],alpha = 1, linestyle = linestyle) # Simulation response 
                    # if ii==len(fileName)-1:
                    # plt.subplot(4,3,jj+7) .plot(timeElapsedExp,moleFracDV*flowInDV/(feedMoleFrac[0]*FeedFlow),
                    #     color='#118ab2',alpha=0.2,
                    #     linestyle = '-') # Dead volume simulation response    
                    plt.subplot(4,3,jj+7) .set(xlabel='$t$ [s]', 
                            ylabel='$Q(t)y(t)/QinYin$ [-]') 
                    plt.fill_between(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracSim*flowRateSim/(initMoleFrac[0]*flowRateSim[0]),
                                     moleFracDV*flowInDV/(initMoleFrac[0]*flowRateSim[0]),alpha = 0.1,color = 'red')
                    plt.subplot(4,3,jj+7) .set(xlabel='$t$ [s]', 
                            ylabel='$Q(t)y(t)/QinYin$ [-]', ylim =  [1e-5, 1]) 
                    
            else:
                plt.subplot(4,3,jj+7) .plot(timeElapsedExp,moleFracSim*flowRateSim/(feedMoleFrac[0]*FeedFlow),
                     color=colorsForPlot[ii],label=legendStr,alpha = 1, linestyle = linestyle) # Simulation response    
                    # if ii==len(fileName)-1:
                plt.subplot(4,3,jj+7) .set(xlabel='$t$ [s]', 
                        ylabel='$Q(t)y(t)/QinYin$ [-]', ylim =  [0, 1]) 
            plt.subplot(4,3,jj+7) .plot(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracDV*flowInDV/(feedMoleFrac[0]*FeedFlow),
                color='#118ab2',alpha=0.2,
                linestyle = '-') # Dead volume simulation response    
            plt.fill_between(timeElapsedExp-1e-3*massSorbent/(adsorbentDensity*FeedFlow),moleFracSim*flowRateSim/(feedMoleFrac[0]*FeedFlow),
                             moleFracDV*flowInDV/(feedMoleFrac[0]*FeedFlow),alpha = 0.1,color = 'red')
            
            plt.subplot(4,3,jj+7) .locator_params(axis="x", nbins=4)
            plt.subplot(4,3,7) .legend()
            
            
            # fract loading
            plt.subplot(4,3,jj+10) .plot(timeElapsedExp,fractionalLoading,
                     color=colorsForPlot[ii],alpha = 1, linestyle = linestyle) # Simulation response               
     
            if flagDesorption:
                plt.subplot(4,3,jj+10) .set(xlabel='$t$ [s]', 
                        ylabel='Fractional loading [-]', ylim =  [0, 1]  )
                        # xlim = [0,3000], ylim =  [0, 1])   
            else:
                plt.subplot(4,3,jj+10) .set(xlabel='$t$ [s]', 
                        ylabel='Fractional loading [-]', ylim =  [0, 1] ) 
                        # xlim = [0,3000], ylim =  [0, 1])  
            plt.subplot(4,3,jj+10) .locator_params(axis="x", nbins=4)
            
            if flagDesorption:
                tpync = 1e-3*massSorbent/(adsorbentDensity*flowRateSim[0])
                tads = np.trapz(moleFracSim*flowRateSim/(moleFracSim[0]*flowRateSim[0]),timeElapsedExp+tpync)
                tblank = np.trapz(moleFracDV*flowInDV/(moleFracSim[0]*flowRateSim[0]),timeElapsedExp+tpync)
                
                
                MassBalance = pressureTotal*moleFracSim[0]*flowRateSim[0]/(Rg*temperatureExp[nn]*massSorbent*1e-3)*(tads-tblank-tpync)
                MassBalanceALL = np.hstack((MassBalanceALL,MassBalance))
                
                plt.subplot(4,3,jj+1) .scatter(moleFracSim[0],MassBalance,s=20,color=colorsForPlot[ii])
            else:
                tads = np.trapz(1-moleFracSim*flowRateSim/(feedMoleFrac[0]*FeedFlow),timeElapsedExp)
                tblank = np.trapz(1-moleFracDV*flowInDV/(feedMoleFrac[0]*FeedFlow),timeElapsedExp)
                tpync =1e-3* massSorbent/(adsorbentDensity*FeedFlow)

                MassBalance = pressureTotal*feedMoleFrac[0]*FeedFlow/(Rg*temperatureExp[nn]*massSorbent*1e-3)*(tads-tblank-tpync)
                MassBalanceALL = np.hstack((MassBalanceALL,MassBalance))
                
                plt.subplot(4,3,jj+1) .scatter(feedMoleFrac[0],MassBalance,s=20,color=colorsForPlot[ii])
plt.show()
