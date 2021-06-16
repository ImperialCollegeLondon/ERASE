############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2021
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#
# Purpose:
# Simulates the full ZLC setup. The model calls the simulate ZLC function 
# to simulate the sorption process and the response is fed to the dead 
# volume simulator
#
# Last modified:
# - 2021-06-16, AK: Add temperature dependence to kinetics
# - 2021-06-01, AK: Add temperature as an input
# - 2021-05-29, AK: Add a separate MS dead volume
# - 2021-05-13, AK: Add volumes and density as inputs
# - 2021-04-27, AK: Convert to a function for parameter estimation
# - 2021-04-22, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
############################################################################

def simulateCombinedModel(**kwargs):
    from simulateZLC import simulateZLC
    from deadVolumeWrapper import deadVolumeWrapper
    from numpy import load
    import os
    import numpy as np

    # Move to top level folder (to avoid path issues)    
    os.chdir("..")
    import auxiliaryFunctions    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    os.chdir("experimental")
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Plot flag
    plotFlag = False
    
    # Isotherm model parameters  (SSL or DSL)
    if 'isothermModel' in kwargs:
        isothermModel = kwargs["isothermModel"]
    else:
        # Default isotherm model is DSL and uses CO2 isotherm on AC8
        # Reference: 10.1007/s10450-020-00268-7
        isothermModel = [0.44, 3.17e-6, 28.63e3, 6.10, 3.21e-6, 20.37e3]

    # Kinetic rate constants [/s]
    if 'rateConstant' in kwargs:
        rateConstant = kwargs["rateConstant"]
    else:
        rateConstant = [0.3]
    
    # Kinetic activation energy constants [J/mol]
    if 'kineticActEnergy' in kwargs:
        kineticActEnergy = np.array(kwargs["kineticActEnergy"])
    else:
        kineticActEnergy = np.array([0]) # To simulate the case with temp. dep.

    # Temperature of the gas [K]
    if 'temperature' in kwargs:
        temperature = np.array(kwargs["temperature"]);
    else:
        temperature = np.array([298.15]);

    # Feed flow rate [m3/s]
    if 'flowIn' in kwargs:
        flowIn = kwargs["flowIn"]
    else:
        flowIn = [5e-7]

    # Initial Gas Mole Fraction [-]
    if 'initMoleFrac' in kwargs:
        initMoleFrac = kwargs["initMoleFrac"]
    else:
        initMoleFrac = [1.]
        
    # Time span for integration [tuple with t0 and tf]
    if 'timeInt' in kwargs:
        timeInt = kwargs["timeInt"]
    else:
        timeInt = (0.0,300)  
        
    # Volume of sorbent material [m3]
    if 'volSorbent' in kwargs:
        volSorbent = kwargs["volSorbent"]
    else:
        volSorbent = 4.35e-8
        
    # Volume of gas chamber (dead volume) [m3]
    if 'volGas' in kwargs:
        volGas = kwargs["volGas"]
    else:
        volGas = 6.81e-8
        
    # Adsorbent density [kg/m3]
    # This has to be the skeletal density
    if 'adsorbentDensity' in kwargs:
        adsorbentDensity = kwargs["adsorbentDensity"]
    else:
        adsorbentDensity = 1950 # Activated carbon skeletal density [kg/m3]
        
    # File name with dead volume characteristics parameters
    if 'deadVolumeFile' in kwargs:
        deadVolumeFile = kwargs["deadVolumeFile"]
    else:
        deadVolumeFile = 'deadVolumeCharacteristics_20210511_1015_ebb447e.npz'  
    
    # Flag to check if experimental data used
    if 'expFlag' in kwargs:
        expFlag = kwargs["expFlag"]
    else:
        expFlag = False

    # Call the simulateZLC function to simulate the sorption in a given sorbent
    timeZLC, resultMat, _ = simulateZLC(isothermModel=isothermModel,
                                        rateConstant=rateConstant,
                                        kineticActEnergy = kineticActEnergy,
                                        temperature = temperature,
                                        flowIn = flowIn,
                                        initMoleFrac = initMoleFrac,
                                        timeInt = timeInt,
                                        expFlag=expFlag,
                                        volSorbent = volSorbent,
                                        volGas = volGas,
                                        adsorbentDensity = adsorbentDensity)
    
    # Parse out the mole fraction out from ZLC
    moleFracZLC = resultMat[0,:]
    
    # Parse out the flow rate out from ZLC [m3/s]
    flowRateZLC = resultMat[3,:]*1e6 # Convert to ccs
    
    # File with parameter estimates for the dead volume
    deadVolumeDir = '..' + os.path.sep + 'simulationResults/'
    modelOutputTemp = load(deadVolumeDir+deadVolumeFile, allow_pickle=True)["modelOutput"]
    # Parse out dead volume parameters
    x = modelOutputTemp[()]["variable"]

    # Get the MS dead volume file, if available
    # Additionally, load the flag that will be used in deadVolumeWrapper
    # This is back compatible with older versions without MS model
    dvFileLoadTemp = load(deadVolumeDir+deadVolumeFile)
    # With MS Model
    if 'flagMSDeadVolume' in dvFileLoadTemp.files:
        flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
        msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
    # Without MS Model
    else:
        flagMSDeadVolume = False
        msDeadVolumeFile = []

    # Call the deadVolume Wrapper function to obtain the outlet mole fraction
    moleFracOut = deadVolumeWrapper(timeZLC, flowRateZLC, x, flagMSDeadVolume, msDeadVolumeFile,
                                    initMoleFrac = moleFracZLC[0], feedMoleFrac = moleFracZLC)
    
    # Plot results if needed
    if plotFlag:
        plotCombinedModel(timeZLC,moleFracOut,moleFracZLC,flowRateZLC)
    
    # Return the time, mole fraction (all), resultMat (ZLC)
    return timeZLC, moleFracOut, resultMat
   
# fun: plotCombinedModel 
# Plots the response of the combined ZLC+DV model
def plotCombinedModel(timeZLC,moleFracOut,moleFracZLC,flowRateZLC):
    import os
    import numpy as np
    os.chdir(".."+os.path.sep+"plotFunctions")
    import matplotlib.pyplot as plt
    
    # Plot the model response
    # Linear scale
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)        
    ax1.plot(timeZLC,moleFracZLC,linewidth = 2,color='b',label='ZLC') # ZLC response
    ax1.plot(timeZLC,moleFracOut,linewidth = 2,color='r',label='ZLC+DV') # Combined model response
    ax1.set(xlabel='$t$ [s]', 
            ylabel='$y_1$ [-]',
            xlim = [0,300], ylim = [0, 1])   
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)      
    ax1.legend()
    
    # Log scale
    ax2 = plt.subplot(1,3,2)   
    ax2.plot(timeZLC,moleFracZLC,linewidth = 2,color='b') # ZLC response       
    ax2.plot(timeZLC,moleFracOut,linewidth = 2,color='r') # Combined model response    
    ax2.set(xlabel='$t$ [s]', 
            xlim = [0,300], ylim = [1e-4, 1.])         
    ax2.locator_params(axis="x", nbins=4)
    ax2.legend()

    # Ft - Log scale
    ax3 = plt.subplot(1,3,3)   
    ax3.semilogy(np.multiply(flowRateZLC,timeZLC),moleFracZLC,linewidth = 2,color='b') # ZLC response       
    ax3.semilogy(np.multiply(flowRateZLC,timeZLC),moleFracOut,linewidth = 2,color='r') # Combined model response    
    ax3.set(xlabel='$t$ [s]', 
            xlim = [0,300], ylim = [1e-4, 1.])         
    ax3.locator_params(axis="x", nbins=4)
    ax3.legend()
    plt.show()
    os.chdir("..")
