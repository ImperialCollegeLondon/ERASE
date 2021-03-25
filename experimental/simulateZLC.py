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
    from simulateSensorArray import simulateSensorArray
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
    
    # Sensor ID
    if 'sensorID' in kwargs:
        sensorID = np.array([kwargs["sensorID"]])
    else:
        sensorID = np.array([6])

    # Kinetic rate constants [/s]
    if 'rateConstant' in kwargs:
        rateConstant = np.array(kwargs["rateConstant"])
    else:
        rateConstant = np.array([.01,.01,.01])

    # Feed flow rate [m3/s]
    if 'flowIn' in kwargs:
        flowIn = np.array(kwargs["flowIn"])
    else:
        flowIn = np.array([5e-7])
    
    # Feed Mole Fraction [-]
    if 'feedMoleFrac' in kwargs:
        feedMoleFrac = np.array(kwargs["feedMoleFrac"])
    else:
        feedMoleFrac = np.array([1.,0.,0.])

    # Initial Gas Mole Fraction [-]
    if 'initMoleFrac' in kwargs:
        initMoleFrac = np.array(kwargs["initMoleFrac"])
    else:
        # Equilibrium process
        initMoleFrac = np.array([0.,0.,1.])

    # Time span for integration [tuple with t0 and tf]
    if 'timeInt' in kwargs:
        timeInt = kwargs["timeInt"]
    else:
        timeInt = (0.0,2000)
                    
    # Volume of sorbent material [m3]
    if 'volSorbent' in kwargs:
        volSorbent = kwargs["volSorbent"]
    else:
        volSorbent = 5e-7
        
    # Volume of gas chamber (dead volume) [m3]
    if 'volGas' in kwargs:
        volGas = kwargs["volGas"]
    else:
        volGas = 5e-7
        
    # Total volume of the system [m3]
    if 'volTotal' in kwargs:
        volTotal = kwargs["volTotal"]
        if 'voidFrac' in kwargs:
            voidFrac = kwargs["voidFrac"]
        else:
            raise Exception("You should provide void fraction if you provide total volume!")
        volGas = voidFrac * volTotal # Volume of gas chamber (dead volume) [m3]
        volSorbent = (1-voidFrac) * volTotal # Volume of solid sorbent [m3]

    if (len(feedMoleFrac) != len(initMoleFrac) 
        or len(feedMoleFrac) != len(rateConstant)):
        raise Exception("The dimensions of the mole fraction or rate constant and the number of gases in the adsorbent is not consistent!")
    else:
        numberOfGases = len(feedMoleFrac)

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
    sensorLoadingPerGasVol, adsorbentDensity, molecularWeight  = simulateSensorArray(sensorID, pressureTotal,
                                                                                     temperature, np.array([initMoleFrac]),
                                                                                     fullModel = True)
    
    # Prepare tuple of input parameters for the ode solver
    inputParameters = (sensorID, adsorbentDensity, rateConstant, numberOfGases, flowIn, 
                       feedMoleFrac, initMoleFrac, pressureTotal, temperature, volSorbent, volGas)
            
    # Solve the system of ordinary differential equations
    # Stiff solver used for the problem: BDF or Radau
    # The output is print out every 5 s
    # Solves the model assuming constant/negligible pressure across the sensor
    # Prepare initial conditions vector
    initialConditions = np.zeros([2*numberOfGases-1])
    initialConditions[0:numberOfGases-1] = initMoleFrac[0:numberOfGases-1] # Gas mole fraction
    initialConditions[numberOfGases-1:2*numberOfGases-1] = sensorLoadingPerGasVol # Initial Loading
    outputSol = solve_ivp(solveSorptionEquationConstP, timeInt, initialConditions, 
                          method='Radau', t_eval = np.arange(timeInt[0],timeInt[1],5),
                          rtol = 1e-6, args = inputParameters)
    
    # Presure vector in output
    pressureVec =  pressureTotal * np.ones(len(outputSol.t)) # Constant pressure

    # Compute the outlet flow rate
    dqdt = np.gradient(outputSol.y[numberOfGases-1:2*numberOfGases-1,:],
                       outputSol.t, axis=1) # Compute gradient of loading
    sum_dqdt = np.sum(dqdt, axis=0) # Time resolved sum of gradient
    flowOut = flowIn - ((volSorbent*(8.314*temperature)/pressureTotal)*(sum_dqdt))
    
    # Parse out the output matrix and add flow rate
    resultMat = np.row_stack((outputSol.y,pressureVec,flowOut))

    # Parse out the time
    timeSim = outputSol.t
    
    # Compute the time resolved sensor response
    sensorFingerPrint = np.zeros([len(timeSim)])
    for ii in range(len(timeSim)):
        loadingTemp = resultMat[numberOfGases-1:2*numberOfGases-1,ii]
        sensorFingerPrint[ii] = np.dot(loadingTemp,molecularWeight)/adsorbentDensity
        
    # Call the plotting function
    if plotFlag:
        plotFullModelResult(timeSim, resultMat, sensorFingerPrint, inputParameters,
                            gitCommitID, currentDT)
    
    # Return time and the output matrix
    return timeSim, resultMat, sensorFingerPrint, inputParameters

# func: solveSorptionEquationConstP - Constant pressure model
# Solves the system of ODEs to evaluate the gas composition and loadings
def solveSorptionEquationConstP(t, f, *inputParameters):  
    import numpy as np
    from simulateSensorArray import simulateSensorArray

    # Gas constant
    Rg = 8.314; # [J/mol K]
    
    # Unpack the tuple of input parameters used to solve equations
    sensorID, _ , rateConstant, numberOfGases, flowIn, feedMoleFrac, _ , pressureTotal, temperature, volSorbent, volGas = inputParameters

    # Initialize the derivatives to zero
    df = np.zeros([2*numberOfGases-1])
    
    # Compute the equilbirium loading at the current gas composition
    currentGasComposition = np.concatenate((f[0:numberOfGases-1],
                                            np.array([1.-np.sum(f[0:numberOfGases-1])])))
    sensorLoadingPerGasVol, _ , _ = simulateSensorArray(sensorID, pressureTotal,
                                                        temperature, np.array([currentGasComposition]),
                                                        fullModel = True)
    
    # Linear driving force model (derivative of solid phase loadings)
    df[numberOfGases-1:2*numberOfGases-1] = np.multiply(rateConstant,(sensorLoadingPerGasVol-f[numberOfGases-1:2*numberOfGases-1]))

    # Total mass balance
    # Assumes constant pressure, so flow rate evalauted
    flowOut = flowIn - ((volSorbent*(Rg*temperature)/pressureTotal)*(np.sum(df[numberOfGases-1:2*numberOfGases-1])))
    
    # Component mass balance
    term1 = 1/volGas
    for ii in range(numberOfGases-1):
        term2 = ((flowIn*feedMoleFrac[ii] - flowOut*f[ii])
                 - (volSorbent*(Rg*temperature)/pressureTotal)*df[ii+numberOfGases-1])
        df[ii] = term1*term2
    
    # Return the derivatives for the solver
    return df

# func: plotFullModelResult
# Plots the model output for the conditions simulated locally
def plotFullModelResult(timeSim, resultMat, sensorFingerPrint, inputParameters,
                        gitCommitID, currentDT):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    # Save settings
    saveFlag = False
    saveFileExtension = ".png"
    
    # Unpack the tuple of input parameters used to solve equations
    sensorID, _ , _ , _ , flowIn, _ , _ , _ , temperature, _ , _ = inputParameters

    os.chdir("plotFunctions")
    if resultMat.shape[0] == 7:
        # Plot the solid phase compositions
        plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
        fig = plt.figure        
        ax = plt.subplot(1,3,1)
        ax.plot(timeSim, resultMat[2,:],
                 linewidth=1.5,color='r')
        ax.set(xlabel='$t$ [s]', 
               ylabel='$q_1$ [mol m$^{\mathregular{-3}}$]',
               xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(resultMat[2,:])])
        
        ax = plt.subplot(1,3,2)
        ax.plot(timeSim, resultMat[3,:],
                 linewidth=1.5,color='b')
        ax.set(xlabel='$t$ [s]', 
           ylabel='$q_2$ [mol m$^{\mathregular{-3}}$]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(resultMat[3,:])])
        
        ax = plt.subplot(1,3,3)
        ax.plot(timeSim, resultMat[4,:],
                 linewidth=1.5,color='g')
        ax.set(xlabel='$t$ [s]', 
           ylabel='$q_3$ [mol m$^{\mathregular{-3}}$]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(resultMat[4,:])])
        
        #  Save the figure
        if saveFlag:
            # FileName: solidLoadingFM_<sensorID>_<currentDateTime>_<gitCommitID>
            saveFileName = "solidLoadingFM_" + str(sensorID) + "_" + currentDT + "_" + gitCommitID + saveFileExtension
            savePath = os.path.join('..','simulationFigures',saveFileName.replace('[','').replace(']',''))
            # Check if inputResources directory exists or not. If not, create the folder
            if not os.path.exists(os.path.join('..','simulationFigures')):
                os.mkdir(os.path.join('..','simulationFigures'))
            plt.savefig (savePath)
            
        plt.show()
            
        # Plot the pressure drop, the flow rate, and the mole fraction
        plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
        fig = plt.figure
        ax = plt.subplot(1,3,1)
        ax.plot(timeSim, resultMat[5,:],
                 linewidth=1.5,color='r')
        ax.set_xlabel('$t$ [s]') 
        ax.set_ylabel('$P$ [Pa]', color='r')
        ax.tick_params(axis='y', labelcolor='r')
        ax.set(xlim = [timeSim[0], timeSim[-1]], 
               ylim = [0, 1.1*np.max(resultMat[5,:])])

        ax2 = plt.twinx()
        ax2.plot(timeSim, resultMat[6,:],
                 linewidth=1.5,color='b')
        ax2.set_ylabel('$F$ [m$^{\mathregular{3}}$ s$^{\mathregular{-1}}$]', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set(xlim = [timeSim[0], timeSim[-1]], 
               ylim = [0, 1.1*np.max(resultMat[6,:])])
        
        ax = plt.subplot(1,3,2)
        ax.plot(timeSim, resultMat[5,:]*resultMat[6,:]/(temperature*8.314),
                 linewidth=1.5,color='k')
        ax.plot(timeSim, resultMat[5,:]*resultMat[6,:]*resultMat[0,:]/(temperature*8.314),
                 linewidth=1.5,color='r')
        ax.plot(timeSim, resultMat[5,:]*resultMat[6,:]*resultMat[1,:]/(temperature*8.314),
                 linewidth=1.5,color='b')
        ax.plot(timeSim, resultMat[5,:]*resultMat[6,:]*(1-resultMat[0,:]-resultMat[1,:])/(temperature*8.314),
                 linewidth=1.5,color='g')
        ax.set(xlabel='$t$ [s]', 
           ylabel='$Q$ [mol s$^{\mathregular{-1}}$]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(resultMat[5,:])*np.max(resultMat[6,:])
                                                     /temperature/8.314])
           
        ax = plt.subplot(1,3,3)
        ax.plot(timeSim, resultMat[0,:],linewidth=1.5,color='r')
        ax.plot(timeSim, resultMat[1,:],linewidth=1.5,color='b')        
        ax.plot(timeSim, 1-resultMat[0,:]-resultMat[1,:],linewidth=1.5,color='g')
        ax.set(xlabel='$t$ [s]', 
           ylabel='$y$ [-]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.])
        plt.show()
        
        # Plot the sensor finger print
        plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
        fig = plt.figure
        ax = plt.subplot(1,1,1)
        ax.plot(timeSim, sensorFingerPrint,
                 linewidth=1.5,color='k')
        ax.set(xlabel='$t$ [s]', 
           ylabel='$m_i$ [g kg$^{\mathregular{-1}}$]',
           xlim = [timeSim[0], timeSim[-1]], ylim = [0, 1.1*np.max(sensorFingerPrint)])
        #  Save the figure
        if saveFlag:
            # FileName: SensorFingerPrintFM_<sensorID>_<currentDateTime>_<gitCommitID>
            saveFileName = "SensorFingerPrintFM_" + str(sensorID) + "_" + currentDT + "_" + gitCommitID + saveFileExtension
            savePath = os.path.join('..','simulationFigures',saveFileName.replace('[','').replace(']',''))
            # Check if inputResources directory exists or not. If not, create the folder
            if not os.path.exists(os.path.join('..','simulationFigures')):
                os.mkdir(os.path.join('..','simulationFigures'))
            plt.savefig (savePath)
        plt.show()

    os.chdir("..")