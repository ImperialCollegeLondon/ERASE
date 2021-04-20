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
        deadVolume_1D = 1.5
    # Dead Volume of the second volume (mixing) [cc]
    if 'deadVolume_2M' in kwargs:
        deadVolume_2M = kwargs["deadVolume_2M"]
    else:
        deadVolume_2M = 2
    # Dead Volume of the second volume (diffusive) [cc]
    if 'deadVolume_2D' in kwargs:
        deadVolume_2D = kwargs["deadVolume_2D"]
    else:
        deadVolume_2D = 0.5
                
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
    # Number of tanks of the second volume (mixing) [-]
    if 'numTanks_2M' in kwargs:
        numTanks_2M = kwargs["numTanks_2M"]
    else:
        numTanks_2M = 1
    # Number of tanks of the second volume (mixing) [-]
    if 'numTanks_2D' in kwargs:
        numTanks_2D = kwargs["numTanks_2D"]
    else:
        numTanks_2D = 1
    
    # Split ratio for flow rate of the first volume [-]
    if 'splitRatio_1' in kwargs:
        splitRatio_1 = kwargs["splitRatio_1"]
    else:
        splitRatio_1 = 0.99
    # Split ratio for flow rate of the second volume [-]
    if 'splitRatio_2' in kwargs:
        splitRatio_2 = kwargs["splitRatio_2"]
    else:
        splitRatio_2 = 0.9
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
                       deadVolume_2M,deadVolume_2D,
                       numTanks_1M, numTanks_1D, numTanks_2M,
                       numTanks_2D, splitRatio_1, splitRatio_2,
                       feedMoleFrac)
    
    # Total number of tanks[-]
    numTanksTotal = numTanks_1M + numTanks_2M + numTanks_1D + numTanks_2D   

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
    moleFracMix = outputSol.y[numTanksTotal-numTanks_2D-1]
    # Diffusive volume
    moleFracDiff = outputSol.y[-1]

    # Composition after mixing
    moleFracOut = np.divide(splitRatio_2*np.multiply(flowRate,moleFracMix)
                    + (1-splitRatio_2)*np.multiply(flowRate,moleFracDiff),flowRate)
    
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
    timeElapsed, flowRateALL, deadVolume_1M, deadVolume_1D, deadVolume_2M, deadVolume_2D, numTanks_1M, numTanks_1D, numTanks_2M, numTanks_2D, splitRatio_1, splitRatio_2, feedMoleFrac = inputParameters

    # Check if experimental data available
    # If size of florate is one, then no need for interpolation
    # If one, then interpolate flow rate values to get at ode time
    if flowRateALL.size != 1:
        interpFlow = interp1d(timeElapsed, flowRateALL)
        flowRate = interpFlow(t)
    else:
        flowRate = flowRateALL
        
    # Total number of tanks [-]
    numTanksTotal = numTanks_1M + numTanks_2M + numTanks_1D + numTanks_2D   

    # Total number of tanks of individual volumes [-]
    numTanksTotal_1 = numTanks_1M + numTanks_1D

    # Initialize the derivatives to zero
    df = np.zeros([numTanksTotal])

    # Volume of each tank in each section
    volTank_1M = deadVolume_1M/numTanks_1M
    volTank_1D = deadVolume_1D/numTanks_1D
    volTank_2M = deadVolume_2M/numTanks_2M
    volTank_2D = deadVolume_2D/numTanks_2D
    
    # Residence time of each tank in the mixing and diffusive volume
    residenceTime_1M = volTank_1M/(splitRatio_1*flowRate)
    residenceTime_1D = volTank_1D/((1-splitRatio_1)*flowRate)
    residenceTime_2M = volTank_2M/(splitRatio_2*flowRate)
    residenceTime_2D = volTank_2D/((1-splitRatio_2)*flowRate)
    
    # Solve the odes
    # Volume 1: Mixing volume
    df[0] = ((1/residenceTime_1M)*(feedMoleFrac - f[0]))
    df[1:numTanks_1M] = ((1/residenceTime_1M)
                         *(f[0:numTanks_1M-1] - f[1:numTanks_1M]))
    
    # Volume 1: Diffusive volume    
    df[numTanks_1M] = ((1/residenceTime_1D)*(feedMoleFrac - f[numTanks_1M]))
    df[numTanks_1M+1:numTanksTotal_1] = ((1/residenceTime_1D)
                                       *(f[numTanks_1M:numTanksTotal_1-1] 
                                         - f[numTanks_1M+1:numTanksTotal_1]))
    
    # Compute the outlet composition for volume 1
    yOut_1 = (splitRatio_1*flowRate*f[numTanks_1M-1] 
                + (1-splitRatio_1)*flowRate*f[numTanksTotal_1-1])/flowRate

    # Volume 2: Mixing volume    
    df[numTanksTotal_1] = ((1/residenceTime_2M)*(yOut_1 - f[numTanksTotal_1]))
    df[numTanksTotal_1+1:numTanks_2M] = ((1/residenceTime_2M)
                                       *(f[numTanksTotal_1:numTanks_2M-1] 
                                         - f[numTanksTotal_1+1:numTanks_2M]))

    # Volume 2: Diffusive volume    
    df[numTanksTotal_1+numTanks_2D] = ((1/residenceTime_2D)
                                       *(yOut_1 - f[numTanksTotal_1+numTanks_2D]))
    df[numTanksTotal_1+numTanks_2D+1:numTanksTotal] = ((1/residenceTime_2D)
                                       *(f[numTanksTotal_1+numTanks_2D:numTanksTotal-1] 
                                         - f[numTanksTotal_1+numTanks_2D+1:numTanksTotal]))

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