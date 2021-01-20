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
# Simulates the sensor chamber as a CSTR incorporating kinetic effects
#
# Last modified:
# - 2021-01-19, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def simulateFullModel(**kwargs):
    import numpy as np
    from scipy.integrate import solve_ivp
    from simulateSensorArray import simulateSensorArray   
    import matplotlib.pyplot as plt
    import os
    
    # Sensor ID
    if 'sensorID' in kwargs:
        sensorID = np.array(kwargs["sensorID"])
    else:
        sensorID = np.array([6])

    # Kinetic rate constants [/s]
    if 'rateConstant' in kwargs:
        rateConstant = np.array(kwargs["rateConstant"])
    else:
        # Equilibrium process
        rateConstant = np.array([0.01,0.01])

    # Feed flow rate [m3/s]
    if 'flowIn' in kwargs:
        flowIn = np.array(kwargs["flowIn"])
    else:
        # Equilibrium process
        flowIn = np.array([5e-7])
    
    # Feed Mole Fraction [-]
    if 'feedMoleFrac' in kwargs:
        feedMoleFrac = np.array(kwargs["feedMoleFrac"])
    else:
        # Equilibrium process
        feedMoleFrac = np.array([1.,0.])

    # Initial Gas Mole Fraction [-]
    if 'initMoleFrac' in kwargs:
        initMoleFrac = np.array(kwargs["initMoleFrac"])
    else:
        # Equilibrium process
        initMoleFrac = np.array([0.,1.])

    if (len(feedMoleFrac) != len(initMoleFrac) 
        or len(feedMoleFrac) != len(rateConstant)):
        raise Exception("The dimensions of the mole fraction or rate constant and the number of gases in the adsorbent is not consistent!")
    else:
        numberOfGases = len(feedMoleFrac)

    # Time span for integration [s]
    timeInt = (0.0,1000)

    # Total pressure of the gas [Pa]
    pressureTotal = np.array([1.e5]);
    
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    temperature = np.array([298.15]);
    
    # Volume of sorbent material [m3]
    volSorbent = 5e-7
        
    # Volume of gas chamber (dead volume) [m3]
    volGas = 5e-7
        
        
    # Compute the initial sensor loading [mol/m3] @ initMoleFrac
    _ , sensorLoadingPerGasVol = simulateSensorArray(sensorID, pressureTotal,
                                                    temperature, np.array([initMoleFrac]),
                                                    fullModel = True)
    
    inputParameters = (sensorID, rateConstant, numberOfGases, flowIn, feedMoleFrac,
                       pressureTotal, temperature, volSorbent, volGas)
        
    # Prepare initial conditions vector
    initialConditions = np.zeros([2*numberOfGases])
    initialConditions[0:numberOfGases-1] = initMoleFrac[0:numberOfGases-1] # Gas mole fraction
    initialConditions[numberOfGases-1:2*numberOfGases-1] = sensorLoadingPerGasVol # Initial Loading
    initialConditions[2*numberOfGases-1] = pressureTotal # Outlet pressure the same as inlet pressure
    
    # Solve the system of ordinary differential equations
    outputSol = solve_ivp(solveSorptionEquation, timeInt, initialConditions, method='BDF', args = inputParameters)


    os.chdir("plotFunctions")
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure
    ax = plt.subplot(1,3,1)
    ax.plot(outputSol.t, outputSol.y[0,:],
             linewidth=1.5,color='r')
    ax.set(xlabel='$t$ [s]', 
           ylabel='$y$ [-]',
           xlim = timeInt, ylim = [0, 1.1*np.max(outputSol.y[0,:])])
    ax = plt.subplot(1,3,2)
    ax.plot(outputSol.t, outputSol.y[1,:],
             linewidth=1.5,color='r')
    ax.set(xlabel='$t$ [s]', 
           ylabel='$q_1$ [mol m$^{\mathregular{-3}}$]',
           xlim = timeInt, ylim = [0, 1.1*np.max(outputSol.y[1,:])])
    ax = plt.subplot(1,3,3)
    ax.plot(outputSol.t, outputSol.y[2,:],
             linewidth=1.5,color='b')
    ax.set(xlabel='$t$ [s]', 
       ylabel='$q_1$ [mol m$^{\mathregular{-3}}$]',
       xlim = timeInt, ylim = [0, 1.1*np.max(outputSol.y[2,:])])
    plt.show()
    
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure
    ax = plt.subplot(1,2,1)
    ax.plot(outputSol.t, outputSol.y[3,:],
             linewidth=1.5,color='r')
    ax.set(xlabel='$t$ [s]', 
       ylabel='$P$ [Pa]',
       xlim = timeInt, ylim = [0, 1.1*np.max(outputSol.y[3,:])])

    ax = plt.subplot(1,2,2)
    ax.plot(outputSol.t, outputSol.y[3,:]*(flowIn/temperature/8.314),
             linewidth=1.5,color='k')
    ax.plot(outputSol.t, outputSol.y[3,:]*outputSol.y[0,:]*(flowIn/temperature/8.314),
             linewidth=1.5,color='r')
    ax.plot(outputSol.t, outputSol.y[3,:]*(1.-outputSol.y[0,:])*(flowIn/temperature/8.314),
             linewidth=1.5,color='b')
    ax.set(xlabel='$t$ [s]', 
       ylabel='$Q$ [mol s$^{\mathregular{-1}}$]',
       xlim = timeInt, ylim = [0, 1.1*np.max(outputSol.y[3,:])*(flowIn/temperature/8.314)])
    plt.show()
    
    os.chdir("..")
    return outputSol

def solveSorptionEquation(t, f, *inputParameters):  
    import numpy as np
    from simulateSensorArray import simulateSensorArray

    # Gas constant
    Rg = 8.314; # [J/mol K]
    
    # Unpack the tuple of input parameters zsed to solve equations
    sensorID, rateConstant, numberOfGases, flowIn, feedMoleFrac, pressureTotal, temperature, volSorbent, volGas = inputParameters

    # Initialize the derivatives to zero
    df = np.zeros([2*numberOfGases])
    
    # Compute the equilbirium loading at the current gas composition
    currentGasComposition = np.concatenate((f[0:numberOfGases-1],
                                            np.array([1.-np.sum(f[0:numberOfGases-1])])))
    _ , sensorLoadingPerGasVol = simulateSensorArray(sensorID, f[2*numberOfGases-1],
                                                    temperature, np.array([currentGasComposition]),
                                                    fullModel = True)
    
    # Linear driving force model (derivative of solid phase loadings)
    df[numberOfGases-1:2*numberOfGases-1] = np.multiply(rateConstant,(sensorLoadingPerGasVol-f[numberOfGases-1:2*numberOfGases-1]))

    # Total mass balance
    term1 = 1/volGas
    term2 = ((flowIn*pressureTotal) - (flowIn*f[2*numberOfGases-1])
             - ((volSorbent*(Rg*temperature))*(np.sum(df[numberOfGases-1:2*numberOfGases-1]))))
    df[2*numberOfGases-1] = term1*term2
    
    # Component mass balance
    term1 = 1/volGas
    for ii in range(numberOfGases-1):
        term2 = (flowIn*(pressureTotal*feedMoleFrac[ii] - f[2*numberOfGases-1]*f[ii])
                 - (volSorbent*(Rg*temperature))*df[ii+numberOfGases-1])
        df[ii] = (term1*term2 - f[ii]*df[2*numberOfGases-1])/f[2*numberOfGases-1]

    return df