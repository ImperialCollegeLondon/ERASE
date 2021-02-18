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
# Simulates a volumetric system that can be used to estimate pure component
# isotherm and kinetics
#
# Last modified:
# - 2021-02-18, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def simulateVolumetricSystem(**kwargs):
    import numpy as np
    from scipy.integrate import solve_ivp
    import auxiliaryFunctions
    
    # Plot flag
    plotFlag = False
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Sensor ID
    if 'sensorID' in kwargs:
        sensorID = np.array([kwargs["sensorID"]])
    else:
        sensorID = np.array([6])
    
    # Time span for integration [tuple with t0 and tf]
    if 'timeInt' in kwargs:
        timeInt = kwargs["timeInt"]
    else:
        timeInt = (0.0,2000)

    # Kinetic rate constant [/s] - Only one constant (pure gas)
    if 'rateConstant' in kwargs:
        rateConstant = np.array(kwargs["rateConstant"])
    else:
        rateConstant = np.array([.001])
        
    # Valve rate constant [mol/s/Pa] - Only one constant (pure gas)
    if 'valveConstant' in kwargs:
        valveConstant = np.array(kwargs["valveConstant"])
    else:
        valveConstant = np.array([0.04e5]) # From 10.1021/la026451+ (different units)
        
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    if 'temperature' in kwargs:
        temperature = np.array(kwargs["temperature"]);
    else:
        temperature = np.array([298.15]);

    # Dosing volume pressure [Pa]
    if 'pressDose' in kwargs:
        pressDose = np.array(kwargs["pressDose"]);
    else:
        pressDose = np.array([1e5]);

    # Uptake volume pressure [Pa]
    if 'pressUptake' in kwargs:
        pressUptake = np.array(kwargs["pressUptake"]);
    else:
        pressUptake = np.array([0.95e5]);
        
    # Dosing volume of the setup [m3]
    if 'volDose' in kwargs:
        volDose = np.array(kwargs["volDose"]);
    else:
        volDose = np.array([385e-6]); # From 10.1021/la026451+ (different units)
        
    # Uptake volume of the setup [m3]
    if 'volUptake' in kwargs:
        volUptake = np.array(kwargs["volUptake"]);
    else:
        volUptake = np.array([366e-6]); # From 10.1021/la026451+ (different units)

    # Uptake voidFrac of the uptake volume [-]
    if 'voidFrac' in kwargs:
        voidFrac = np.array(kwargs["voidFrac"]);
    else:
        voidFrac = np.array([0.98]); # From 10.1021/la026451+ (calculated)
            
    # Initial Gas Mole Fraction [-]
    # Only pure gas system for volumetric system (DO NOT CHANGE)
    initMoleFrac = np.array([1.,0.])
    
    # Number of gases - Note this is only for loading purposes 
    # Only pure gas is simulated
    # Used to chose the isotherm properties from the inputResources folder
    numberOfGases = len(initMoleFrac)

    # Prepare tuple of input parameters for the ode solver
    inputParameters = (sensorID, rateConstant, valveConstant,
                       numberOfGases, initMoleFrac, temperature, voidFrac,
                       volDose, volUptake)
            
    # Initial conditions for the solver
    initialConditions = np.zeros([4])
    initialConditions[0] = pressDose # Dosing pressure [Pa]
    initialConditions[1] = pressUptake # Uptake pressure [Pa]
    # 2 and 3 are solid loading and the moles through the valve 
    
    # Solve the system of ordinary differential equations
    # Stiff solver used for the problem: BDF or Radau
    outputSol = solve_ivp(solveVolumetricSystemEquation, timeInt, initialConditions, 
                      method='Radau', t_eval = np.arange(timeInt[0],timeInt[1],1),
                      args = inputParameters)
    
    # Parse out the time and the pressure and loading from the output
    timeSim = outputSol.t
    resultMat = outputSol.y
    
    # Call the plotting function
    if plotFlag:
        plotFullModelResult(timeSim, resultMat, gitCommitID, currentDT)
    
    # Return time, the output matrix, and the parameters used in the simulation
    return timeSim, resultMat, inputParameters

# func: solveVolumetricSystemEquation
# Solves the system of ODEs to simulate the volumetric system
def solveVolumetricSystemEquation(t, f, *inputParameters):
    import numpy as np
    from simulateSensorArray import simulateSensorArray

    # Gas constant
    Rg = 8.314; # [J/mol K]

    # Unpack the tuple of input parameters used to solve equations
    sensorID, rateConstant, valveConstant, numberOfGases, initMoleFrac, temperature, voidFrac, volDose, volUptake = inputParameters

    # Initialize the derivatives to zero
    df = np.zeros([4])
    
    sensorLoadingPerGasVol = simulateSensorArray(sensorID, f[1],
                                                        temperature, np.array([initMoleFrac]))
    
    # Linear driving force model (derivative of solid phase loadings)
    df[2] = rateConstant*(sensorLoadingPerGasVol[0]-f[2])

    # Valve flow model (derivative of gas flow through the valve)
    df[3] = valveConstant*(f[0]-f[1])
    
    # Rate of change of pressure in the dosing volume
    df[0] = -(Rg*temperature/volDose)*df[3]
    
    # Rate of change of pressure in the uptake volume
    df[1] = (Rg*temperature/(volUptake*voidFrac))*(df[3] - (1-voidFrac)*volUptake*df[2])
    
    # Return the derivatives for the solver
    return df
    
# func: plotFullModelResult
# Plots the model output for the conditions simulated locally
def plotFullModelResult(timeSim, resultMat, gitCommitID, currentDT):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    # Save settings
    saveFlag = False
    saveFileExtension = ".png"

    os.chdir("plotFunctions")
    # Plot the solid phase compositions
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure        
    ax = plt.subplot(1,2,1)
    ax.plot(timeSim, resultMat[0,:]/resultMat[0,0],
             linewidth=1.5,color='r', label = 'Dosing')
    ax.plot(timeSim, resultMat[1,:]/resultMat[0,0],
             linewidth=1.5,color='b', label = 'Uptake')
    ax.set(xlabel='$t$ [s]', 
           ylabel='$P/P_0$ [-]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0., 1.])
    
    ax = plt.subplot(1,2,2)
    ax.plot(timeSim, resultMat[2,:],
             linewidth=1.5,color='k')
    ax.set(xlabel='$t$ [s]', 
       ylabel='$q$ [mol m$^{\mathregular{-3}}$]',
       xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(resultMat[2,:])])
        
    #  Save the figure
    # if saveFlag:
        # # FileName: solidLoadingFM_<sensorID>_<currentDateTime>_<gitCommitID>
        # saveFileName = "solidLoadingFM_" + str(sensorID) + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        # savePath = os.path.join('..','simulationFigures',saveFileName.replace('[','').replace(']',''))
        # # Check if inputResources directory exists or not. If not, create the folder
        # if not os.path.exists(os.path.join('..','simulationFigures')):
        #     os.mkdir(os.path.join('..','simulationFigures'))
        # plt.savefig (savePath)
        
    plt.show()

    os.chdir("..")