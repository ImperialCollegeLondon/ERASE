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
# Simulates the ZLC setup. This is inspired from Ruthven's work and from the 
# sensor model. Note that there is no analytical solution and it uses a full 
# model with mass transfer defined using linear driving force.
#
# Last modified:
# - 2021-04-26, AK: Revamp the code for real sorbent simulation
# - 2021-03-25, AK: Remove the constant F model
# - 2021-03-01, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
############################################################################

def simulateZLC(**kwargs):
    import numpy as np
    from scipy.integrate import solve_ivp
    from computeEquilibriumLoading import computeEquilibriumLoading
    import auxiliaryFunctions
    import os
    
    # Move to top level folder (to avoid path issues)
    os.chdir("..")
    
    # Plot flag
    plotFlag = False
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Kinetic rate constants [/s]
    if 'rateConstant' in kwargs:
        rateConstant = np.array(kwargs["rateConstant"])
    else:
        rateConstant = np.array([0.1])

    # Feed flow rate [m3/s]
    if 'flowIn' in kwargs:
        flowIn = np.array(kwargs["flowIn"])
    else:
        flowIn = np.array([5e-7])
    
    # Feed Mole Fraction [-]
    if 'feedMoleFrac' in kwargs:
        feedMoleFrac = np.array(kwargs["feedMoleFrac"])
    else:
        feedMoleFrac = np.array([0.])

    # Initial Gas Mole Fraction [-]
    if 'initMoleFrac' in kwargs:
        initMoleFrac = np.array(kwargs["initMoleFrac"])
    else:
        # Equilibrium process
        initMoleFrac = np.array([1.])

    # Time span for integration [tuple with t0 and tf]
    if 'timeInt' in kwargs:
        timeInt = kwargs["timeInt"]
    else:
        timeInt = (0.0,300)
                    
    # Volume of sorbent material [m3]
    if 'volSorbent' in kwargs:
        volSorbent = kwargs["volSorbent"]
    else:
        volSorbent = 1.5e-8
        
    # Volume of gas chamber (dead volume) [m3]
    if 'volGas' in kwargs:
        volGas = kwargs["volGas"]
    else:
        volGas = 2.5e-8
        
    # Adsorbent density [kg/m3]
    if 'adsorbentDensity' in kwargs:
        adsorbentDensity = kwargs["adsorbentDensity"]
    else:
        adsorbentDensity = 760 # Activated carbon density [kg/m3]

    # Total pressure of the gas [Pa]
    if 'pressureTotal' in kwargs:
        pressureTotal = np.array(kwargs["pressureTotal"]);
    else:
        pressureTotal = np.array([1.e5]);
        
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    if 'temperature' in kwargs:
        temperature = np.array(kwargs["temperature"]);
    else:
        temperature = np.array([298.15]);
        
    # Compute the initial sensor loading [mol/m3] @ initMoleFrac
    equilibriumLoading  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                    temperature=temperature,
                                                    moleFrac=initMoleFrac)*adsorbentDensity # [mol/m3]
    
    # Prepare tuple of input parameters for the ode solver
    inputParameters = (adsorbentDensity, rateConstant, flowIn, feedMoleFrac, 
                       initMoleFrac, pressureTotal, temperature, volSorbent, volGas)
            
    # Solve the system of ordinary differential equations
    # Stiff solver used for the problem: BDF or Radau
    # The output is print out every 0.1 s
    # Solves the model assuming constant/negligible pressure across the sensor
    # Prepare initial conditions vector
    initialConditions = np.zeros([2])
    initialConditions[0] = initMoleFrac[0] # Gas mole fraction
    initialConditions[1] = equilibriumLoading # Initial Loading

    outputSol = solve_ivp(solveSorptionEquation, timeInt, initialConditions, 
                          method='Radau', t_eval = np.arange(timeInt[0],timeInt[1],0.1),
                          rtol = 1e-8, args = inputParameters)
    
    # Presure vector in output
    pressureVec =  pressureTotal * np.ones(len(outputSol.t)) # Constant pressure

    # Compute the outlet flow rate
    sum_dqdt = np.gradient(outputSol.y[1,:],
                       outputSol.t) # Compute gradient of loading
    flowOut = flowIn - ((volSorbent*(8.314*temperature)/pressureTotal)*(sum_dqdt))
    
    # Parse out the output matrix and add flow rate
    resultMat = np.row_stack((outputSol.y,pressureVec,flowOut))

    # Parse out the time
    timeSim = outputSol.t
    
    # Call the plotting function
    if plotFlag:
        plotFullModelResult(timeSim, resultMat, inputParameters,
                            gitCommitID, currentDT)
    
    # Return time and the output matrix
    return timeSim, resultMat, inputParameters

# func: solveSorptionEquation - Constant pressure model
# Solves the system of ODEs to evaluate the gas composition and loadings
def solveSorptionEquation(t, f, *inputParameters):  
    import numpy as np
    from computeEquilibriumLoading import computeEquilibriumLoading
    
    # Gas constant
    Rg = 8.314; # [J/mol K]

    # Unpack the tuple of input parameters used to solve equations
    adsorbentDensity, rateConstant, flowIn, feedMoleFrac, _ , pressureTotal, temperature, volSorbent, volGas = inputParameters

    # Initialize the derivatives to zero
    df = np.zeros([2])
    
    # Compute the initial sensor loading [mol/m3] @ initMoleFrac
    equilibriumLoading  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                    temperature=temperature,
                                                    moleFrac=f[0])*adsorbentDensity # [mol/m3]

    # Linear driving force model (derivative of solid phase loadings)
    df[1] = rateConstant*(equilibriumLoading-f[1])

    # Total mass balance
    # Assumes constant pressure, so flow rate evalauted
    flowOut = flowIn - (volSorbent*(Rg*temperature)/pressureTotal)*df[1]
    
    # Component mass balance
    term1 = 1/volGas
    term2 = ((flowIn*feedMoleFrac - flowOut*f[0])
             - (volSorbent*(Rg*temperature)/pressureTotal)*df[1])
    df[0] = term1*term2
    
    # Return the derivatives for the solver
    return df

# func: plotFullModelResult
# Plots the model output for the conditions simulated locally
def plotFullModelResult(timeSim, resultMat, inputParameters,
                        gitCommitID, currentDT):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    # Save settings
    saveFlag = False
    saveFileExtension = ".png"
    
    # Unpack the tuple of input parameters used to solve equations
    adsorbentDensity , _ , flowIn, _ , _ , _ , temperature, _ , _ = inputParameters

    os.chdir("plotFunctions")
    # Plot the solid phase compositions
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    fig = plt.figure        
    ax = plt.subplot(1,3,1)
    ax.plot(timeSim, resultMat[1,:]/adsorbentDensity,
             linewidth=1.5,color='r')
    ax.set(xlabel='$t$ [s]', 
           ylabel='$q_1$ [mol kg$^{\mathregular{-1}}$]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(resultMat[1,:]/adsorbentDensity)])
    
    ax = plt.subplot(1,3,2)
    ax.semilogy(timeSim, resultMat[0,:],linewidth=1.5,color='r')
    ax.set(xlabel='$t$ [s]', 
       ylabel='$y$ [-]',
       xlim = [timeSim[0], timeSim[-1]], ylim = [1e-4, 1.])
    
    ax = plt.subplot(1,3,3)
    ax.plot(timeSim, resultMat[2,:],
             linewidth=1.5,color='r')
    ax.set_xlabel('$t$ [s]') 
    ax.set_ylabel('$P$ [Pa]', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.set(xlim = [timeSim[0], timeSim[-1]], 
           ylim = [0, 1.1*np.max(resultMat[2,:])])

    ax2 = plt.twinx()
    ax2.plot(timeSim, resultMat[3,:],
             linewidth=1.5,color='b')
    ax2.set_ylabel('$F$ [m$^{\mathregular{3}}$ s$^{\mathregular{-1}}$]', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set(xlim = [timeSim[0], timeSim[-1]], 
           ylim = [0, 1.1*np.max(resultMat[3,:])])
    
    #  Save the figure
    if saveFlag:
        # FileName: ZLCResponse_<currentDateTime>_<gitCommitID>
        saveFileName = "ZLCResponse_" + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures',saveFileName.replace('[','').replace(']',''))
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures')):
            os.mkdir(os.path.join('..','simulationFigures'))
        plt.savefig (savePath)
    plt.show()

    os.chdir("..")