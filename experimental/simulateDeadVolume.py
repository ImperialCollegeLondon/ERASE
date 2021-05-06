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
# The methodolgy is slighlty modified to incorporate diffusive pockets using
# compartment models (see Levenspiel, chapter 12 or Lisa Joss's article)
# Reference: 10.1007/s10450-012-9417-z
#
# Last modified:
# - 2021-05-03, AK: Fix path issues
# - 2021-04-26, AK: Change default model parameter values
# - 2021-04-21, AK: Change model to fix split velocity
# - 2021-04-20, AK: Change model to flow dependent split
# - 2021-04-20, AK: Change model to flow dependent split
# - 2021-04-20, AK: Implement time-resolved experimental flow rate for DV
# - 2021-04-14, AK: Change from simple TIS to series of parallel CSTRs
# - 2021-04-12, AK: Small fixed
# - 2021-03-25, AK: Fix for plot
# - 2021-03-18, AK: Fix for inlet concentration
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
    import os
    
    # Move to top level folder (to avoid path issues)
    os.chdir("..")
    
    # Plot flag
    plotFlag = False
        
    # Flow rate of the gas [cc/s]
    if 'flowRate' in kwargs:
        flowRate = kwargs["flowRate"]
    else:
        flowRate = np.array([0.25])
    # Dead Volume of the first volume [cc]
    if 'deadVolume_1' in kwargs:
        deadVolume_1 = kwargs["deadVolume_1"]
    else:
        deadVolume_1 = 4.25
    # Number of tanks of the first volume [-]
    if 'numTanks_1' in kwargs:
        numTanks_1 = kwargs["numTanks_1"]
    else:
        numTanks_1 = 30
    # Dead Volume of the second volume (mixing) [cc]
    if 'deadVolume_2M' in kwargs:
        deadVolume_2M = kwargs["deadVolume_2M"]
    else:
        deadVolume_2M = 1.59 
    # Dead Volume of the second volume (diffusive) [cc]
    if 'deadVolume_2D' in kwargs:
        deadVolume_2D = kwargs["deadVolume_2D"]
    else:
        deadVolume_2D = 5.93e-1
    # Flow rate in the diffusive volume [-]
    if 'flowRate_D' in kwargs:
        flowRate_D = kwargs["flowRate_D"]
    else:
        flowRate_D = 1.35e-2
    # Initial Mole Fraction [-]
    if 'initMoleFrac' in kwargs:
        initMoleFrac = np.array(kwargs["initMoleFrac"])
    else:
        initMoleFrac = np.array([1.])
    # Feed Mole Fraction [-]
    if 'feedMoleFrac' in kwargs:
        feedMoleFrac = np.array(kwargs["feedMoleFrac"])
    else:
        feedMoleFrac = np.array([0.])
    # Time span for integration [tuple with t0 and tf]
    if 'timeInt' in kwargs:
        timeInt = kwargs["timeInt"]
    else:
        timeInt = (0.0,3600)
    
    # Flag to check if experimental data used
    if 'expFlag' in kwargs:
        expFlag = kwargs["expFlag"]
    else:
        expFlag = False
    
    # If experimental data used, then initialize ode evaluation time to 
    # experimental time, else use default
    if expFlag is False:
        t_eval = np.arange(timeInt[0],timeInt[-1],0.1)
    else:
        # Use experimental time (from timeInt) for ode evaluations to avoid
        # interpolating any data. t_eval is also used for interpolating
        # flow rate in the ode equations
        t_eval = timeInt
        timeInt = (0.0,max(timeInt))

    # Prepare tuple of input parameters for the ode solver
    inputParameters = (t_eval,flowRate, deadVolume_1,deadVolume_2M,
                       deadVolume_2D, numTanks_1, flowRate_D,
                       feedMoleFrac)
    
    # Total number of tanks[-]
    numTanksTotal = numTanks_1 + 2

    # Prepare initial conditions vector
    # The first element is the inlet composition and the rest is the dead 
    # volume
    initialConditions = np.ones([numTanksTotal])*initMoleFrac
    # Solve the system of equations
    outputSol = solve_ivp(solveTanksInSeries, timeInt, initialConditions, 
                          method='Radau', t_eval = t_eval,
                          rtol = 1e-8, args = inputParameters)
    
    # Parse out the time
    timeSim = outputSol.t
    
    # Inlet concentration
    moleFracIn = np.ones((len(outputSol.t),1))*feedMoleFrac

    # Mole fraction at the outlet
    # Mixing volume
    moleFracMix = outputSol.y[numTanks_1]
    # Diffusive volume
    moleFracDiff = outputSol.y[-1]

    # Composition after mixing
    flowRate_M = flowRate - flowRate_D
    moleFracOut = np.divide(np.multiply(flowRate_M,moleFracMix)
                    + np.multiply(flowRate_D,moleFracDiff),flowRate)

    # Plot the dead volume response
    if plotFlag:
        plotOutletConcentration(timeSim,moleFracIn,moleFracOut)
        
    # Move to local folder (to avoid path issues)
    os.chdir("experimental")

    return timeSim, moleFracIn, moleFracOut

# func: solveTanksInSeries
# Solves the system of ODE for the tanks in series model for the dead volume        
def solveTanksInSeries(t, f, *inputParameters):
    import numpy as np
    from scipy.interpolate import interp1d

    # Unpack the tuple of input parameters used to solve equations
    timeElapsed, flowRateALL, deadVolume_1, deadVolume_2M, deadVolume_2D, numTanks_1, flowRate_D, feedMoleFracALL = inputParameters

    # Check if experimental data available
    # If size of flowrate is one, then no need for interpolation
    # If one, then interpolate flow rate values to get at ode time
    if flowRateALL.size != 1:
        interpFlow = interp1d(timeElapsed, flowRateALL)
        flowRate = interpFlow(t)
    else:
        flowRate = flowRateALL
        
    # If size of mole fraction is one, then no need for interpolation
    # If one, then interpolate mole fraction values to get at ode time
    if feedMoleFracALL.size != 1:
        interpMoleFrac = interp1d(timeElapsed, feedMoleFracALL)
        feedMoleFrac = interpMoleFrac(t) 
    else:
        feedMoleFrac = feedMoleFracALL
        
    # Total number of tanks [-]
    numTanksTotal = numTanks_1 + 2

   # Initialize the derivatives to zero
    df = np.zeros([numTanksTotal])

    # Volume 1: Mixing volume
    # Volume of each tank in the mixing volume
    volTank_1 = deadVolume_1/numTanks_1
    residenceTime_1 = volTank_1/(flowRate)
    
    # Solve the odes
    df[0] = ((1/residenceTime_1)*(feedMoleFrac - f[0]))
    df[1:numTanks_1] = ((1/residenceTime_1)
                         *(f[0:numTanks_1-1] - f[1:numTanks_1]))
  
    # Volume 2: Diffusive volume
    # Volume of each tank in the mixing volume
    volTank_2M = deadVolume_2M
    volTank_2D = deadVolume_2D
    
    # Residence time of each tank in the mixing and diffusive volume
    flowRate_M = flowRate - flowRate_D
    residenceTime_2M = volTank_2M/(flowRate_M)
    residenceTime_2D = volTank_2D/(flowRate_D)
    
    # Solve the odes
    # Volume 2: Mixing volume
    df[numTanks_1] = ((1/residenceTime_2M)*(f[numTanks_1-1] - f[numTanks_1]))
    
    # Volume 2: Diffusive volume    
    df[numTanks_1+1] = ((1/residenceTime_2D)*(f[numTanks_1-1] - f[numTanks_1+1]))

    # Return the derivatives for the solver
    return df

# func: plotOutletConcentration
# Plot the concentration outlet after correcting for dead volume
def plotOutletConcentration(timeSim, moleFracIn, moleFracOut):
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # Plot the solid phase compositions
    os.chdir("plotFunctions")
    plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure        
    ax = plt.subplot(1,1,1)
    ax.plot(timeSim, moleFracIn,
             linewidth=1.5,color='b',
             label = 'In')
    ax.semilogy(timeSim, moleFracOut,
             linewidth=1.5,color='r',
             label = 'Out')
    ax.set(xlabel='$t$ [s]', 
           ylabel='$y$ [-]',
           xlim = [timeSim[0], 1000], ylim = [1e-4, 1.1*np.max(moleFracOut)])
    ax.legend()
    plt.show()
    os.chdir("..")