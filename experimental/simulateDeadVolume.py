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
# Simulates the dead volume using the tanks in series (TIS) for the ZLC
# Reference: 10.1016/j.ces.2008.02.023
#
# Last modified:
# - 2021-03-17, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def simulateDeadVolume(**kwargs):
    import numpy as np
    from scipy.integrate import solve_ivp
    from simulateSensorArray import simulateSensorArray
    import auxiliaryFunctions
    
    # Plot flag
    plotFlag = False
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Flow rate of the gas [cc/s]
    if 'flowRate' in kwargs:
        flowRate = kwargs["flowRate"]
    else:
        flowRate = 0.25
    # Total Dead Volume of the tanks [cc]
    if 'deadVolume' in kwargs:
        deadVolume = kwargs["deadVolume"]
    else:
        deadVolume = 1
    # Number of tanks  [-]
    if 'numberOfTanks' in kwargs:
        numberOfTanks = kwargs["numberOfTanks"]
    else:
        numberOfTanks = 2
    # Total pressure of the gas [Pa]
    if 'pressureTotal' in kwargs:
        pressureTotal = np.array(kwargs["pressureTotal"]);
    else:
        pressureTotal = np.array([1.e5]);
    # Temperature of the gas [K]
    if 'temperature' in kwargs:
        temperature = np.array(kwargs["temperature"]);
    else:
        temperature = np.array([298.15]);
    # Feed Mole Fraction [-]
    if 'feedMoleFrac' in kwargs:
        feedMoleFrac = np.array(kwargs["feedMoleFrac"])
    else:
        feedMoleFrac = np.array([1.])
    # Time span for integration [tuple with t0 and tf]
    if 'timeInt' in kwargs:
        timeInt = kwargs["timeInt"]
    else:
        timeInt = (0.0,20)
    
    # Gas constant
    Rg = 8.314; # [J/mol K]
    
    # Prepare tuple of input parameters for the ode solver
    inputParameters = (flowRate, deadVolume, numberOfTanks, pressureTotal, temperature)            

    # Prepare initial conditions vector
    # The first element is the inlet composition and the rest is the dead 
    # volume
    initialConditions = np.concatenate((feedMoleFrac, 
                                        np.ones([numberOfTanks])*(1-feedMoleFrac)))
    # Solve the system of equations
    outputSol = solve_ivp(solveTanksInSeries, timeInt, initialConditions, 
                          method='Radau', t_eval = np.arange(timeInt[0],timeInt[1],0.1),
                          rtol = 1e-6, args = inputParameters)
    
    # Parse out the time
    timeSim = outputSol.t
    
    # Inlet concentration
    moleFracIn = outputSol.y[0]
    
    # Outlet concentration at the dead volume
    moleFracOut = outputSol.y[-1]
    
    # Plot the dead volume response
    if plotFlag:
        plotOutletConcentration(timeSim,moleFracIn,moleFracOut)

    return timeSim, moleFracIn, moleFracOut

# func: solveTanksInSeries
# Solves the system of ODE for the tanks in series model for the dead volume        
def solveTanksInSeries(t, f, *inputParameters):
    import numpy as np
    
    # Gas constant
    Rg = 8.314; # [J/mol K]
    
    # Unpack the tuple of input parameters used to solve equations
    flowRate, deadVolume , numberOfTanks, pressureTotal, temperature = inputParameters

    # Initialize the derivatives to zero
    df = np.zeros([numberOfTanks+1])

    # Volume of each tank
    volumeOfTank = deadVolume/numberOfTanks
    
    # Residence time of each tank
    residenceTime = volumeOfTank/flowRate

    # Solve the ode
    df[1:numberOfTanks+1] = ((1/residenceTime)
                             *(f[0:numberOfTanks] - f[1:numberOfTanks+1]))
    
    # Return the derivatives for the solver
    return df

# func: plotOutletConcentration
# Plot the concentration outlet after correcting for dead volume
def plotOutletConcentration(timeSim, moleFracIn, moleFracOut):
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # Plot the solid phase compositions
    os.chdir(".."+os.path.sep+"plotFunctions")
    plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure        
    ax = plt.subplot(1,1,1)
    ax.plot(timeSim, moleFracIn,
             linewidth=1.5,color='b',
             label = 'In')
    ax.plot(timeSim, moleFracOut,
             linewidth=1.5,color='r',
             label = 'Out')
    ax.set(xlabel='$t$ [s]', 
           ylabel='$y$ [-]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(moleFracOut)])
    ax.legend()
    plt.show()
    os.chdir(".."+os.path.sep+"experimental")