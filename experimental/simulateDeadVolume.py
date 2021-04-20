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
# compartment models (see Levenspiel, chapter 12) or Lisa Joss's article
# Reference: 10.1007/s10450-012-9417-z
#
# Last modified:
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
        flowRate = np.array([0.25])
    # Dead Volume of the first volume (mixing) [cc]
    if 'deadVolume_1M' in kwargs:
        deadVolume_1M = kwargs["deadVolume_1M"]
    else:
        deadVolume_1M = 5
    # Dead Volume of the first volume (diffusive) [cc]
    if 'deadVolume_1D' in kwargs:
        deadVolume_1D = kwargs["deadVolume_1D"]
    else:
        deadVolume_1D = 1                
    # Number of tanks of the first volume (mixing) [-]
    if 'numTanks_1M' in kwargs:
        numTanks_1M = kwargs["numTanks_1M"]
    else:
        numTanks_1M = 10
    # Number of tanks of the first volume (mixing) [-]
    if 'numTanks_1D' in kwargs:
        numTanks_1D = kwargs["numTanks_1D"]
    else:
        numTanks_1D = 1
    # Split ratio for flow rate of the first volume [-]
    if 'splitRatioFactor' in kwargs:
        splitRatioFactor = kwargs["splitRatioFactor"]
    else:
        splitRatioFactor = 1.1
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
        timeInt = (0.0,2000)
    
    # If experimental data used, then initialize ode evaluation time to 
    # experimental time, else use default
    if flowRate.size == 1:
        t_eval = np.arange(timeInt[0],timeInt[-1],0.1)
    else:
        # Use experimental time (from timeInt) for ode evaluations to avoid
        # interpolating any data. t_eval is also used for interpolating
        # flow rate in the ode equations
        t_eval = timeInt
        timeInt = (0.0,max(timeInt))

    # Prepare tuple of input parameters for the ode solver
    inputParameters = (t_eval,flowRate, deadVolume_1M,deadVolume_1D,
                       numTanks_1M, numTanks_1D, splitRatioFactor,
                       feedMoleFrac)
    
    # Total number of tanks[-]
    numTanksTotal = numTanks_1M + numTanks_1D

    # Prepare initial conditions vector
    # The first element is the inlet composition and the rest is the dead 
    # volume
    initialConditions = np.ones([numTanksTotal])*initMoleFrac
    # Solve the system of equations
    outputSol = solve_ivp(solveTanksInSeries, timeInt, initialConditions, 
                          method='Radau', t_eval = t_eval,
                          rtol = 1e-6, args = inputParameters)
    
    # Parse out the time
    timeSim = outputSol.t
    
    # Inlet concentration
    moleFracIn = np.ones((len(outputSol.t),1))*feedMoleFrac

    # Mole fraction at the outlet
    # Mixing volume
    moleFracMix = outputSol.y[numTanks_1M-1]
    # Diffusive volume
    moleFracDiff = outputSol.y[-1]

    # Composition after mixing
    splitRatio_1 = np.divide(np.multiply(splitRatioFactor,flowRate),
                             (1+np.multiply(splitRatioFactor,flowRate)))
    moleFracOut = np.divide(np.multiply(splitRatio_1,np.multiply(flowRate,moleFracMix))
                    + np.multiply((1-splitRatio_1),np.multiply(flowRate,moleFracDiff)),flowRate)

    # Plot the dead volume response
    if plotFlag:
        plotOutletConcentration(timeSim,moleFracIn,moleFracOut)

    return timeSim, moleFracIn, moleFracOut

# func: solveTanksInSeries
# Solves the system of ODE for the tanks in series model for the dead volume        
def solveTanksInSeries(t, f, *inputParameters):
    import numpy as np
    from scipy.interpolate import interp1d

    # Unpack the tuple of input parameters used to solve equations
    timeElapsed, flowRateALL, deadVolume_1M, deadVolume_1D, numTanks_1M, numTanks_1D, splitRatioFactor, feedMoleFracALL = inputParameters

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
    numTanksTotal = numTanks_1M + numTanks_1D

    # Initialize the derivatives to zero
    df = np.zeros([numTanksTotal])

    # Volume of each tank in each section
    volTank_1M = deadVolume_1M/numTanks_1M
    volTank_1D = deadVolume_1D/numTanks_1D
    
    # Residence time of each tank in the mixing and diffusive volume
    splitRatio_1 = splitRatioFactor*flowRate/(1+splitRatioFactor*flowRate)
    residenceTime_1M = volTank_1M/(splitRatio_1*flowRate)
    residenceTime_1D = volTank_1D/((1-splitRatio_1)*flowRate)
    
    # Solve the odes
    # Volume 1: Mixing volume
    df[0] = ((1/residenceTime_1M)*(feedMoleFrac - f[0]))
    df[1:numTanks_1M] = ((1/residenceTime_1M)
                         *(f[0:numTanks_1M-1] - f[1:numTanks_1M]))
    
    # Volume 1: Diffusive volume    
    df[numTanks_1M] = ((1/residenceTime_1D)*(feedMoleFrac - f[numTanks_1M]))
    df[numTanks_1M+1:numTanksTotal] = ((1/residenceTime_1D)
                                       *(f[numTanks_1M:numTanksTotal-1] 
                                         - f[numTanks_1M+1:numTanksTotal]))
    
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
    ax.semilogy(timeSim, moleFracOut,
             linewidth=1.5,color='r',
             label = 'Out')
    ax.set(xlabel='$t$ [s]', 
           ylabel='$y$ [-]',
           xlim = [timeSim[0], 1000], ylim = [1e-4, 1.1*np.max(moleFracOut)])
    ax.legend()
    plt.show()
    os.chdir(".."+os.path.sep+"experimental")