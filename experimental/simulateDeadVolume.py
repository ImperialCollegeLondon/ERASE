############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2021
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#           Hassan Azzan (HA)
#           Tristan Spreng (TS)
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
    from scipy.integrate import odeint, solve_ivp
    import os
    import pdb
    
    # Move to top level folder (to avoid path issues)
    os.chdir("..")
    
    # Plot flag
    plotFlag = False
        
    ### VARIABLES (set before)
    #  Flow rate of the gas [cc/s]
    if 'flowRate' in kwargs:
        flowRate = kwargs["flowRate"]
    else:
        flowRate = np.array([0.25])
        
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
    
    ### PARAMETERS (determined by fitting), numbers indicating position in matrix
    # 0. Dead Volume of the first volume [cc]
    if 'deadVolume_1' in kwargs:
        deadVolume_1 = kwargs["deadVolume_1"]
    else:
        deadVolume_1 = 4.25
        
    # 1. Dead Volume of the second volume (mixing) [cc]
    if 'deadVolume_2M' in kwargs:
        deadVolume_2M = kwargs["deadVolume_2M"]
    else:
        deadVolume_2M = 1.59 
        
    # 2. Dead Volume of the second volume (diffusive) [cc]
    if 'deadVolume_2D' in kwargs:
        deadVolume_2D = kwargs["deadVolume_2D"]
    else:
        deadVolume_2D = 5.93e-1
    
    # 3. Number of tanks of the first volume [-]
    if 'numTanks_1' in kwargs:
        numTanks_1 = kwargs["numTanks_1"]
    else:
        numTanks_1 = 30
   
    # 4. Flow rate in the diffusive volume [-]
    if 'flowRate_2D' in kwargs:
        flowRate_2D = kwargs["flowRate_2D"]
    else:
        flowRate_2D = 1.35e-2
    
    # 5. Dead Volume of the third volume (mixing 2) [cc]
    if 'deadVolume_2M_2' in kwargs:
        deadVolume_2M_2 = kwargs["deadVolume_2M_2"]
    else:
        deadVolume_2M_2 = 0
    
    # 6. Dead Volume of the fourth volume (diffusive 2) [cc]
    if 'deadVolume_2D_2' in kwargs:
        deadVolume_2D_2 = kwargs["deadVolume_2D_2"]
    else:
        deadVolume_2D_2 = 0
    
    # 7. Flow rate in the diffusive volume 2 [-]
    if 'flowRate_2D_2' in kwargs:
        flowRate_2D_2 = kwargs["flowRate_2D_2"]
    else:
        flowRate_2D_2 = 0

    # 8. Dead Volume of the third volume (mixing 2) [cc]
    if 'deadVolume_2M_3' in kwargs:
        deadVolume_2M_3 = kwargs["deadVolume_2M_3"]
    else:
        deadVolume_2M_3 = 0
    
    # 9. Dead Volume of the fourth volume (diffusive 2) [cc]
    if 'deadVolume_2D_3' in kwargs:
        deadVolume_2D_3 = kwargs["deadVolume_2D_3"]
    else:
        deadVolume_2D_3 = 0
    
    # 10. Flow rate in the diffusive volume 2 [-]
    if 'flowRate_2D_3' in kwargs:
        flowRate_2D_3 = kwargs["flowRate_2D_3"]
    else:
        flowRate_2D_3 = 0
    
    
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
                       deadVolume_2D, numTanks_1, flowRate_2D,
                       feedMoleFrac, deadVolume_2M_2,deadVolume_2D_2, flowRate_2D_2, 
                       deadVolume_2M_3,deadVolume_2D_3, flowRate_2D_3)
    
    # Specify total number of tanks[-] depending on model type used
    if flowRate_2D_2 == 0:
        numTanksTotal = numTanks_1 + 2
    elif flowRate_2D_3 == 0:
        numTanksTotal = numTanks_1 + 4
    else:
        numTanksTotal = numTanks_1 + 6

    # Prepare initial conditions vector
    # The first element is the inlet composition and the rest is the dead 
    # volume
    initialConditions = np.ones([numTanksTotal])*initMoleFrac
    # pdb.set_trace()
    ##########################################################################
    # Solve the system of equations
    outputSol = solve_ivp(solveTanksInSeries, timeInt, initialConditions, 
                          method='Radau', t_eval = t_eval,
                          rtol = 1e-8, args = inputParameters, first_step =0.0001, dense_output=True)
    
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
    flowRate_M = flowRate - flowRate_2D
    moleFracOut = np.divide(np.multiply(flowRate_M,moleFracMix)
                    + np.multiply(flowRate_2D,moleFracDiff),flowRate)
    
    
    ##########################################################################
    # # Solve the system of equations
    # outputSol = odeint(solveTanksInSeries, initialConditions, t_eval,
    #                       inputParameters, tfirst=True, h0 = 0.0001,hmax = 0.001)
    
    # # Parse out the time
    # timeSim = t_eval
    
    # # Inlet concentration
    # moleFracIn = np.ones((len(t_eval),1))*feedMoleFrac

    # # Mole fraction at the outlet
    # # Mixing volume
    # moleFracMix = outputSol[numTanks_1,1]
    # # Diffusive volume
    # moleFracDiff = outputSol[-1,1]

    # # Composition after mixing
    # flowRate_M = flowRate - flowRate_2D
    # moleFracOut = np.divide(np.multiply(flowRate_M,moleFracMix)
    #                 + np.multiply(flowRate_2D,moleFracDiff),flowRate)
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
    import pdb
    # Unpack the tuple of input parameters used to solve equations
    timeElapsed, flowRateALL, deadVolume_1, deadVolume_2M, deadVolume_2D, numTanks_1, flowRate_2D, feedMoleFracALL, deadVolume_2M_2, deadVolume_2D_2, flowRate_2D_2, deadVolume_2M_3, deadVolume_2D_3, flowRate_2D_3 = inputParameters

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

    # Total number of tanks[-]
    if flowRate_2D_2 == 0:
        numTanksTotal = numTanks_1 + 2
    elif flowRate_2D_3 == 0: 
        numTanksTotal = numTanks_1 + 4
    else:
        numTanksTotal = numTanks_1 + 6


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
    flowRate_M = flowRate - flowRate_2D
    residenceTime_2M = volTank_2M/(flowRate_M)
    residenceTime_2D = volTank_2D/(flowRate_2D)
    
    # Solve the odes
    # Volume 2: Mixing volume
    df[numTanks_1] = ((1/residenceTime_2M)*(f[numTanks_1-1] - f[numTanks_1]))
    
    # Volume 2: Diffusive volume    
    df[numTanks_1+1] = ((1/residenceTime_2D)*(f[numTanks_1-1] - f[numTanks_1+1]))
    
    ###################### ADD 2ND SET OF PARALLEL REACTORS #########################
    # Total number of tanks[-]
    if flowRate_2D_2 == 0:
        1+1
    else:
        # Mole fraction at the outlet
        # Mixing volume
        moleFracMix = f[numTanks_1]
        # Diffusive volume
        moleFracDiff = f[numTanks_1+1]
    
        # Composition after mixing
        flowRate_M = flowRate - flowRate_2D_2
        moleFracOut = np.divide(np.multiply(flowRate_M,moleFracMix)
                        + np.multiply(flowRate_2D_2,moleFracDiff),flowRate)
        
        # Volume 2: Diffusive volume 2
        # Volume of each tank in the mixing volume 2
        volTank_2M2 = deadVolume_2M_2
        volTank_2D2 = deadVolume_2D_2
        
        # Residence time of each tank in the mixing and diffusive volumes 2
        flowRate_M = flowRate - flowRate_2D_2
        residenceTime_2M2 = volTank_2M2/(flowRate_M)
        
        # Solve the odes
        # Volume 2: Mixing volume 2
        df[numTanks_1+2] = ((1/residenceTime_2M2)*(moleFracOut - f[numTanks_1+2])) ## NOT SURE ABOUT NUMBERS ADDED (esp. moleFracOut)
        
        # Volume 2: Diffusive volume 2  
        if flowRate_2D_2 == 0:
            df[numTanks_1+3] = 0
            # pdb.set_trace()
        else:
            residenceTime_2D2= volTank_2D2/(flowRate_2D_2)
            df[numTanks_1+3] = ((1/residenceTime_2D2)*(moleFracOut - f[numTanks_1+3]))

    ###################### ADD 3RD SET OF PARALLEL REACTORS #########################
    # Total number of tanks[-]
    if flowRate_2D_3 == 0:
        1+1
    else:
        # Mole fraction at the outlet
        # Mixing volume
        moleFracMix = f[numTanks_1]
        # Diffusive volume
        moleFracDiff = f[numTanks_1+1]
    
        # Composition after mixing
        flowRate_M = flowRate - flowRate_2D_3
        moleFracOut = np.divide(np.multiply(flowRate_M,moleFracMix)
                        + np.multiply(flowRate_2D_3,moleFracDiff),flowRate)
        
        # Volume 2: Diffusive volume 2
        # Volume of each tank in the mixing volume 2
        volTank_2M3 = deadVolume_2M_3
        volTank_2D3 = deadVolume_2D_3
        
        # Residence time of each tank in the mixing and diffusive volumes 2
        flowRate_M = flowRate - flowRate_2D_3
        residenceTime_2M3 = volTank_2M3/(flowRate_M)
        
        # Solve the odes
        # Volume 2: Mixing volume 2
        df[numTanks_1+4] = ((1/residenceTime_2M3)*(moleFracOut - f[numTanks_1+4]))
        
        # Volume 2: Diffusive volume 2  
        if flowRate_2D_3 == 0:
            df[numTanks_1+5] = 0
            # pdb.set_trace()
        else:
            residenceTime_2D3= volTank_2D3/(flowRate_2D_3)
            df[numTanks_1+5] = ((1/residenceTime_2D3)*(moleFracOut - f[numTanks_1+5]))
            
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