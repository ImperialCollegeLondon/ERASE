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
#
# Purpose:
# Simulates the ZLC setup. This is inspired from Ruthven's work and from the 
# sensor model. Note that there is no analytical solution and it uses a full 
# model with mass transfer defined using linear driving force.
#
# Last modified:
# - 2021-08-20, AK: Change definition of rate constants
# - 2021-06-16, AK: Add temperature correction factor to LDF
# - 2021-06-15, AK: Add correction factor to LDF
# - 2021-05-13, AK: IMPORTANT: Change density from particle to skeletal
# - 2021-04-27, AK: Fix inputs and add isotherm model as input
# - 2021-04-26, AK: Revamp the code for real sorbent simulation
# - 2021-03-25, AK: Remove the constant F model
# - 2021-03-01, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def simulateZLC(**kwargs):
    import numpy as np
    from scipy.integrate import odeint, solve_ivp, ode
    from computeEquilibriumLoading import computeEquilibriumLoading
    import auxiliaryFunctions
    import os
    import pdb
    
    # Move to top level folder (to avoid path issues)
    os.chdir("..")
    
    # Plot flag
    plotFlag = False
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Kinetic rate constant 1 (analogous to micropore resistance) [/s]
    if 'rateConstant_1' in kwargs:
        rateConstant_1 = np.array(kwargs["rateConstant_1"])
    else:
        rateConstant_1 = np.array([3.79204704e+03])
        rateConstant_1 = np.array([3*3.79204704e+03])
        
    # Kinetic rate constant 2 (analogous to macropore resistance) [/s]
    if 'rateConstant_2' in kwargs:
        rateConstant_2 = np.array(kwargs["rateConstant_2"])
    else:
        rateConstant_2 = np.array([3.00373911e+01])
        # rateConstant_2 = np.array([0.01])
       
    # Kinetic rate constant 3 [/s]
    if 'rateConstant_3' in kwargs:
        rateConstant_3 = np.array(kwargs["rateConstant_3"])
    else:
        rateConstant_3 = np.array([100])
        # rateConstant_3 = np.array([0.01*0.0025**2])
        
    # Feed flow rate [m3/s]
    if 'flowIn' in kwargs:
        flowIn = np.array(kwargs["flowIn"])
    else:
        flowIn = np.array([10*1.66667e-8])
    
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
        timeInt = (0.0,600)
        
    # Flag to check if experimental data used
    if 'expFlag' in kwargs:
        expFlag = kwargs["expFlag"]
    else:
        expFlag = False

    # If experimental data used, then initialize ode evaluation time to 
    # experimental time, else use default
    if expFlag is False:
        t_eval = np.arange(timeInt[0],timeInt[-1],0.2)
    else:
        # Use experimental time (from timeInt) for ode evaluations to avoid
        # interpolating any data. t_eval is also used for interpolating
        # flow rate in the ode equations
        t_eval = timeInt
        timeInt = (0.0,max(timeInt))
                    
    # Volume of sorbent material [m3]
    if 'volSorbent' in kwargs:
        volSorbent = kwargs["volSorbent"]
    else:
        volSorbent = 4.350000000000000e-08
        
    # Volume of gas chamber (dead volume) [m3]
    if 'volGas' in kwargs:
        volGas = kwargs["volGas"]
    else:
        volGas = 6.803846153846154e-08
        
    # Isotherm model parameters  (SSL or DSL)
    if 'isothermModel' in kwargs:
        isothermModel = kwargs["isothermModel"]
    else:
        # Default isotherm model is DSL and uses CO2 isotherm on AC8
        # Reference: 10.1007/s10450-020-00268-7
        isothermModel = [4.18*3/4,0.5/np.exp(17.7e4/(8.314*298.15)),17.7e4,0,0,0]
        isothermModel = [995*3/4,0.0001/np.exp(2.9e4/(8.314*298.15)),2.9e4,0,0,0]

    # Adsorbent density [kg/m3]
    # This has to be the skeletal density
    if 'adsorbentDensity' in kwargs:
        adsorbentDensity = kwargs["adsorbentDensity"]
    else:
        adsorbentDensity = 2000 # Activated carbon skeletal density [kg/m3]

    # Total pressure of the gas [Pa]
    if 'pressureTotal' in kwargs:
        pressureTotal = np.array(kwargs["pressureTotal"]);
    else:
        pressureTotal = np.array([1e5]);
        
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    if 'temperature' in kwargs:
        temperature = np.array(kwargs["temperature"]);
    else:
        temperature = np.array([298.15]);
        
    # Flag to check model fitting type
    if 'modelType' in kwargs:
        modelType = kwargs["modelType"]
    else:
        modelType = 'KineticSBMacro'
        modelType = 'Diffusion'
        
    # Flag to check mean pore radius
    if 'rpore' in kwargs:
        rpore = kwargs["rpore"]
    else:
        rpore = 1e-9
        
    # Flag to check Dpvals
    if 'Dpvals' in kwargs:
        Dpvals = kwargs["Dpvals"]
    else:
        Dpvals = [3.05671321526166e-05,	3.15050794527196e-05,	3.24331710687508e-05]
        
        
    if modelType == 'Diffusion' or modelType == 'Diffusion1T':
        particleEpsilon = volGas/(volGas+volSorbent)
        inputParameters = (adsorbentDensity, isothermModel, rateConstant_1, rateConstant_2, rateConstant_3,
                           flowIn, feedMoleFrac, initMoleFrac, pressureTotal, 
                           temperature, volSorbent, volGas, modelType)
        tspan, Y, r, yOut, flowOut, qAverage = DiffusionAdsorption1D(initMoleFrac, t_eval, volSorbent, volGas, adsorbentDensity, particleEpsilon, flowIn, temperature, pressureTotal, [isothermModel], rateConstant_1, rateConstant_2, rateConstant_3)
        # Presure vector in output
        pressureVec =  pressureTotal * np.ones(len(tspan)) # Constant pressure

        yOut = np.transpose(yOut)
        flowOut = np.transpose(flowOut)
        # Parse out the output matrix and add flow rate
        resultMat = np.row_stack((yOut,qAverage,pressureVec,flowOut))
        # pdb.set_trace()
        # Parse out the time
        timeSim = tspan
        
    elif modelType == 'Diffusion1TNI':
        particleEpsilon = volGas/(volGas+volSorbent)
        inputParameters = (adsorbentDensity, isothermModel, rateConstant_1, rateConstant_2, rateConstant_3,
                           flowIn, feedMoleFrac, initMoleFrac, pressureTotal, 
                           temperature, volSorbent, volGas, modelType)
        tspan, Y, r, yOut, flowOut, qAverage, tempVals = DiffusionAdsorption1DNI(initMoleFrac, t_eval, volSorbent, volGas, adsorbentDensity, particleEpsilon, flowIn, temperature, pressureTotal, [isothermModel], rateConstant_1, rateConstant_2, rateConstant_3)
        # Presure vector in output
        pressureVec =  pressureTotal * np.ones(len(tspan)) # Constant pressure

        yOut = np.transpose(yOut)
        flowOut = np.transpose(flowOut)
        # Parse out the output matrix and add flow rate
        resultMat = np.row_stack((yOut,qAverage,pressureVec,flowOut))
        # pdb.set_trace()
        # Parse out the time
        timeSim = tspan
    elif modelType == 'Diffusion1Ttau':
        particleEpsilon = volGas/(volGas+volSorbent)
        inputParameters = (adsorbentDensity, isothermModel, rateConstant_1, rateConstant_2, rateConstant_3,
                           flowIn, feedMoleFrac, initMoleFrac, pressureTotal, 
                           temperature, volSorbent, volGas, modelType, rpore, Dpvals)
        tspan, Y, r, yOut, flowOut, qAverage = DiffusionAdsorption1Dtau(initMoleFrac, t_eval, volSorbent, volGas, adsorbentDensity, particleEpsilon, flowIn, temperature, pressureTotal, [isothermModel], rateConstant_1, rateConstant_2, rateConstant_3, rpore, Dpvals)
        # Presure vector in output
        pressureVec =  pressureTotal * np.ones(len(tspan)) # Constant pressure

        yOut = np.transpose(yOut)
        flowOut = np.transpose(flowOut)
        # Parse out the output matrix and add flow rate
        resultMat = np.row_stack((yOut,qAverage,pressureVec,flowOut))
        # pdb.set_trace()
        # Parse out the time
        timeSim = tspan
    elif modelType == 'Diffusion1TNItau':
        particleEpsilon = volGas/(volGas+volSorbent)
        inputParameters = (adsorbentDensity, isothermModel, rateConstant_1, rateConstant_2, rateConstant_3,
                           flowIn, feedMoleFrac, initMoleFrac, pressureTotal, 
                           temperature, volSorbent, volGas, modelType, rpore)
        tspan, Y, r, yOut, flowOut, qAverage, tempVals  = DiffusionAdsorption1DNItau(initMoleFrac, t_eval, volSorbent, volGas, adsorbentDensity, particleEpsilon, flowIn, temperature, pressureTotal, [isothermModel], rateConstant_1, rateConstant_2, rateConstant_3, rpore)
        # Presure vector in output
        pressureVec =  pressureTotal * np.ones(len(tspan)) # Constant pressure

        yOut = np.transpose(yOut)
        flowOut = np.transpose(flowOut)
        # Parse out the output matrix and add flow rate
        resultMat = np.row_stack((yOut,qAverage,pressureVec,flowOut))
        # pdb.set_trace()
        # Parse out the time
        timeSim = tspan
    else:
        # Compute the initial sensor loading [mol/m3] @ initMoleFrac
        equilibriumLoading  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                        temperature=temperature,
                                                        moleFrac=initMoleFrac,
                                                        isothermModel=isothermModel)*adsorbentDensity # [mol/m3]
        
        # Prepare tuple of input parameters for the ode solver
        inputParameters = (adsorbentDensity, isothermModel, rateConstant_1, rateConstant_2, rateConstant_3,
                           flowIn, feedMoleFrac, initMoleFrac, pressureTotal, 
                           temperature, volSorbent, volGas, modelType, rpore)
                
        # Solve the system of ordinary differential equations
        # Stiff solver used for the problem: BDF or Radau
        # The output is print out every 0.1 s
        # Solves the model assuming constant/negligible pressure across the sensor
        # Prepare initial conditions vector
        initialConditions = np.zeros([2])
        initialConditions[0] = initMoleFrac[0] # Gas mole fraction
        initialConditions[1] = equilibriumLoading # Initial Loading
        
        # pdb.set_trace()
        
        ##########################################################################
        outputSol = solve_ivp(solveSorptionEquation, timeInt, initialConditions, 
                              method='Radau', t_eval = t_eval,
                              rtol = 1e-8, args = inputParameters, first_step =0.0001, dense_output=True)
        # Presure vector in output
        pressureVec =  pressureTotal * np.ones(len(outputSol.t)) # Constant pressure
        # pdb.set_trace()
        # Compute the outlet flow rate
        # if outputSol.y[1,:].size != 0:
        sum_dqdt = np.gradient(outputSol.y[1,:],
                            outputSol.t) # Compute gradient of loading
        flowOut = flowIn - ((volSorbent*(8.314*temperature)/pressureTotal)*(sum_dqdt))
        # else:
        #     flowOut = flowIn
        
        # Parse out the output matrix and add flow rate
        resultMat = np.row_stack((outputSol.y,pressureVec,flowOut))
    
        # Parse out the time
        timeSim = outputSol.t
        ##########################################################################
        # ode15s = ode(solveSorptionEquation)
        # ode15s.set_integrator('vode', method='bdf', order=15, nsteps=3000)
        # ode15s.set_initial_value(initialConditions, 0)
        # ode15s.set_f_params(*adsorbentDensity, isothermModel, rateConstant_1, rateConstant_2,
        #                    flowIn, feedMoleFrac, initMoleFrac, pressureTotal, 
        #                    temperature, volSorbent, volGas)
        # pdb.set_trace()
        # ode15s.integrate(timeInt[-1])
        # tVals =  np.arange(timeInt[0],timeInt[-1],0.01)
        # pdb.set_trace()
        # tVals = t_eval
        # outputSol = odeint(solveSorptionEquation, initialConditions, tVals, inputParameters,tfirst=True, h0 = 0.01)
        
        # # Presure vector in output
        # pressureVec =  pressureTotal * np.ones(len(outputSol)) # Constant pressure
    
        # # Compute the outlet flow rate
        # sum_dqdt = np.gradient(outputSol[:,1],
        #                    tVals) # Compute gradient of loading
        # flowOut = flowIn - ((volSorbent*(8.314*temperature)/pressureTotal)*(sum_dqdt))
        
        # # Parse out the output matrix and add flow rate
        # resultMat = np.row_stack((np.transpose(outputSol),pressureVec,flowOut))
    
        # # Parse out the time
        # timeSim = tVals
        # Call the plotting function
    if plotFlag:
        plotFullModelResult(timeSim, resultMat, inputParameters,
                            gitCommitID, currentDT)
    
    # Move to local folder (to avoid path issues)
    os.chdir("experimental")
    # pdb.set_trace()
    # Return time and the output matrix
    return timeSim, resultMat, inputParameters

# func: solveSorptionEquation - Constant pressure model
# Solves the system of ODEs to evaluate the gas composition and loadings
def solveSorptionEquation(t, f, *inputParameters):  
    import numpy as np
    from computeEquilibriumLoading import computeEquilibriumLoading
    # import pdb
    # Gas constant
    Rg = 8.314; # [J/mol K]

    # Unpack the tuple of input parameters used to solve equations
    adsorbentDensity, isothermModel, rateConstant_1, rateConstant_2, rateConstant_3, flowIn, feedMoleFrac, _ , pressureTotal, temperature, volSorbent, volGas, modelType, rpore = inputParameters

    # Initialize the derivatives to zero
    df = np.zeros([2])
    if f[0] < 0:
        f[0] = 0
    
    # Compute the loading [mol/m3] @ f[0]
    equilibriumLoading  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                    temperature=temperature,
                                                    moleFrac=f[0],
                                                    isothermModel=isothermModel)*adsorbentDensity # [mol/m3]
    # Partial pressure of the gas
    partialPressure = f[0]*pressureTotal
    # partialPressure = 0.5*pressureTotal
    # delta pressure to compute gradient
    delP = 10
    # Mole fraction (up)
    moleFractionUp = (partialPressure + delP)/pressureTotal
    # Compute the loading [mol/m3] @ moleFractionUp
    equilibriumLoadingUp  = computeEquilibriumLoading(pressureTotal=pressureTotal,
                                                    temperature=temperature,
                                                    moleFrac=moleFractionUp,
                                                    isothermModel=isothermModel)*adsorbentDensity # [mol/m3]

    
    # Compute the gradient (delq*/dc)
    dqbydc = (equilibriumLoadingUp-equilibriumLoading)/(delP/(Rg*temperature)) # [-]
    dellogp = np.log(partialPressure+delP)-np.log((partialPressure))
    dlnqbydlnp = (np.log(equilibriumLoadingUp)-np.log(equilibriumLoading))/dellogp
    epsilonp = volGas/(volGas+volSorbent)
    
    if modelType == 'KineticOld':
    # Rate constant 1 (analogous to micropore resistance)
        k1 = rateConstant_1
    
        # Rate constant 2 (analogous to macropore resistance)
        k2 = rateConstant_2/dqbydc
        
        # Overall rate constant
        # The following conditions are done for purely numerical reasons
        # If pure (analogous) macropore
        if k1<1e-9:
            rateConstant = k2
        # If pure (analogous) micropore
        elif k2<1e-9:
            rateConstant = k1
        # If both resistances are present
        else:
            rateConstant = 1/(1/k1 + 1/k2)
            
    if modelType == 'Kinetic':
    # Rate constant 1 (analogous to micropore resistance)
        k1 = rateConstant_1/dlnqbydlnp
    
        # Rate constant 2 (analogous to macropore resistance)
        k2 = rateConstant_2/(1+(1/epsilonp)*dqbydc)
        
        # Overall rate constant
        # The following conditions are done for purely numerical reasons
        # If pure (analogous) macropore
        if k1<1e-9:
            rateConstant = k2
        # If pure (analogous) micropore
        elif k2<1e-9:
            rateConstant = k1
        # If both resistances are present
        else:
            rateConstant = 1/(1/k1 + 1/k2)
            
    elif modelType == 'KineticMacro':
        k1 = rateConstant_1/(1+(1/epsilonp)*dqbydc)*np.power(temperature,0.5)
        k2 = rateConstant_2/(1+(1/epsilonp)*dqbydc)*np.power(temperature,1.5)/partialPressure
        if k1<1e-9:
            rateConstant = k2
        # If pure (analogous) micropore
        elif k2<1e-9:
            rateConstant = k1
        # If both resistances are present
        else:
            rateConstant = 1/(1/k1 + 1/k2)
            
    elif modelType == 'KineticSB':
        rateConstant = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))/dlnqbydlnp
        if rateConstant<1e-8:
            rateConstant = 1e-8

    elif modelType == 'KineticSBMacro':
        k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))/dlnqbydlnp
        # k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))
        # k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))
        # Rate constant 2 (analogous to macropore resistance)
        k2 = rateConstant_3*temperature**0.5/(1+(1/epsilonp)*dqbydc)
        k2 = rateConstant_3*np.power(temperature,0.5)*epsilonp/(epsilonp+(1-epsilonp)*dqbydc)        
        k2 = 15*rateConstant_3*(temperature/288.15)**1.75/(epsilonp+(1-epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75*epsilonp/(epsilonp+(1-epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75/(1+(1/epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75*epsilonp/(1+(1/epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75*epsilonp

        # k2 = rateConstant_3
        # k2 = rateConstant_3/(1+(1/epsilonp)*dqbydc)
        if k1<1e-9:
            rateConstant = k2
        # If pure (analogous) micropore
        elif k2<1e-9:
            rateConstant = k1
        # If both resistances are present
        else:
            rateConstant = 1/(1/k1 + 1/k2)
    
        if rateConstant<1e-8:
            rateConstant = 1e-8    
    elif modelType == 'KineticSBMacroTau':
        Rp = ((volSorbent + volGas) / (4/3 * np.pi))**(1/3)  # pellet radius [m]]

        DmolVals = [5.62e-5, 5.95e-5, 6.29e-5]
        # rpore = 107e-9
        Dk = (2/3 * rpore * (8*8.314*temperature/(np.pi*0.044))**0.5)
    
        if temperature == 288.15:
            Dmol = DmolVals[0]
        elif temperature == 298.15:
            Dmol = DmolVals[1]
        elif temperature == 308.15:
            Dmol = DmolVals[2] 
        else:
            Dmol = 0

        Dmac = epsilonp/rateConstant_3*(1/((1/Dk+1/Dmol)))
        k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))/dlnqbydlnp
        # k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))
        # k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))
        # Rate constant 2 (analogous to macropore resistance)
        k2 = rateConstant_3*temperature**0.5/(1+(1/epsilonp)*dqbydc)
        k2 = rateConstant_3*np.power(temperature,0.5)*epsilonp/(epsilonp+(1-epsilonp)*dqbydc)        
        k2 = 15*(Dmac/Rp**2)/(epsilonp+(1-epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75*epsilonp/(epsilonp+(1-epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75/(1+(1/epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75*epsilonp/(1+(1/epsilonp)*dqbydc)
        # k2 = 15*rateConstant_3*(temperature/288.15)**1.75*epsilonp

        # k2 = rateConstant_3
        # k2 = rateConstant_3/(1+(1/epsilonp)*dqbydc)
        if k1<1e-9:
            rateConstant = k2
        # If pure (analogous) micropore
        elif k2<1e-9:
            rateConstant = k1
        # If both resistances are present
        else:
            rateConstant = 1/(1/k2)
    
        if rateConstant<1e-8:
            rateConstant = 1e-8    
            
            
    elif modelType == 'KineticSBMacro2':
        k1 = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))/dlnqbydlnp
        # Rate constant 2 (analogous to macropore resistance)
        tc = 0.01/(rateConstant_3*temperature**0.5*1/(1+(1/epsilonp)*dqbydc))
        k2 = 5.14/(tc*(1+(1/epsilonp)*dqbydc)/(rateConstant_3*temperature**0.5))**0.5
        # pdb.set_trace()
        # k2 = rateConstant_3*np.power(temperature,0.5)*epsilonp/(epsilonp+(1-epsilonp)*dqbydc)
        # k2 = rateConstant_3/(1+(1/epsilonp)*dqbydc)        
        # Overall rate constant
        # The following conditions are done for purely numerical reasons
        # If pure (analogous) macropore
        if k1<1e-9:
            rateConstant = k2
        # If pure (analogous) micropore
        elif k2<1e-9:
            rateConstant = k1
        # If both resistances are present
        else:
            rateConstant = 1/(1/k1 + 1/k2)
    
        if rateConstant<1e-8:
            rateConstant = 1e-8    
            
    # Linear driving force model (derivative of solid phase loadings)
    df[1] = rateConstant*(equilibriumLoading-f[1])
    
    # Quadratic driving force model (derivative of solid phase loadings)
    # df[1] = rateConstant*(equilibriumLoading**2-f[1]**2)/(2*f[1])

    # Total mass balance
    # Assumes constant pressure, so flow rate evalauted
    flowOut = flowIn - (volSorbent*(Rg*temperature)/pressureTotal)*df[1]
    
    # Component mass balance
    term1 = 1/volGas
    term2 = ((flowIn*feedMoleFrac - flowOut*f[0]) - (volSorbent*(Rg*temperature)/pressureTotal)*df[1])
    df[0] = term1*term2
    # pdb.set_trace()

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
    adsorbentDensity, _, _, _, _, flowIn, _, _ , _, temperature, _, _, modelType = inputParameters

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
    
def DiffusionAdsorption1D(Y0, tspan, volSorbent, volGas, adsorbentDensity, epsilon, volFlow, temperature, Ptotal, isothermModelAll, rateConstant_1, rateConstant_2, rateConstant_3):
    import numpy as np
    from scipy.integrate import odeint
    # import pdb
    # import matplotlib.pyplot as plt
    # from numpy import load
    # import os
    # import matplotlib.pyplot as plt
    # import auxiliaryFunctions
    import scipy
    # import nbkode
    # plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    # from scipy.integrate import odeint, solve_ivp, ode

    # Constants and others
    Rg = 8.314  # Universal gas constant [J/molK]
    # pdb.set_trace()
    Rp = ((volSorbent + volGas) / (4/3 * np.pi))**(1/3)  # pellet radius [m]]
    n = 60
    r = np.linspace(0,Rp+2*Rp/n, n)  # discretize radial domain
    
    # pdb.set_trace()
    # Simulate model for input isotherms
    for isothermModel in isothermModelAll:
        # Initial conditions
        c0 = np.zeros(n)  # concentration in macropore [mol/m^3]
        c0[:] = Ptotal * Y0 / (Rg * temperature)  # Initial condition c(r,0) = c0 [mol/m^3]
        c0[-1] = 0
        q0 = computeEquilibrium(c0, temperature, isothermModel, adsorbentDensity)  # Initial condition equilibrium q(r,0) = q*(c,T) [mol/m^3]
        q0[-2:] = 0
        x0 = np.concatenate([c0, q0])    
        inputArgs = r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp

        # ODE solver
        scipy.integrate.ode(radialDiffusionAdsorption1D).set_integrator('vode', method='bdf', order=15)
        # scipy.integrate.ode(radialDiffusionAdsorption1D).set_integrator('lsoda',first_step =0.001)
        Y = odeint(radialDiffusionAdsorption1D, x0, tspan, args=inputArgs)
        
        
        # solver = nbkode.BDF6(radialDiffusionAdsorption1D, 0, x0, params = inputArgs)
        
        # tspan, Y = solver.run(tspan)
        
        moleGas = np.zeros(len(tspan))
        moleSolid = np.zeros(len(tspan))
        qAverage_i = np.zeros(len(tspan))
        for jj in range(len(tspan)):
            qAverage_i[jj] = 3 / (Rp**3) *             np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
            moleGas[jj] = volGas * 3 / (Rp**3) *       np.trapz(Y[jj, 0:n-2]   * r[0:-2] ** 2, r[0:-2])
            moleSolid[jj] = volSorbent * 3 / (Rp**3) * np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
        moleTotal = moleSolid + moleGas
        moleRate = np.gradient(moleTotal, tspan)
        volRate = -moleRate * Rg * temperature / Ptotal
        yOut = Y[:,n-2]*(Rg*temperature)/Ptotal
        volFlowOut = (volRate + volFlow)
        qAverage = qAverage_i
        yOut[yOut < 1e-5] = 1e-5
        volFlowOut[volFlowOut < volFlow] = volFlow
        volFlowOut[volFlowOut > 100*volFlow] = 100*volFlow
        qAverage[qAverage < 1e-6] = 1e-6
        # pdb.set_trace()
    
    return tspan, Y, r, yOut, volFlowOut, qAverage

def radialDiffusionAdsorption1D(x, t, r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp):
    import numpy as np
    # import computedqbydc
    # import computedlnqbydlnp
    # import computeEquilibrium
    import pdb
    Rg = 8.314
    Ptotal = 1e5
    DcDt = np.zeros(n)
    DqDt = np.zeros(n)
    DyDt = np.zeros(n)

    deltar = r[1] - r[0]
    D2cDx2 = np.zeros(n)
    
    c = x[0:n]
    q = x[n:2*n]
    
    # if t == 0:
    #     c[-2] = Y0*Ptotal/(Rg*temperature)
    
    Dmaceff = np.zeros(n)
    kmiceff = np.zeros(n)
    kmic = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))
    # Dmac = rateConstant_3*(Rp**2)*(temperature**0.5)
    Dmac = rateConstant_3*(Rp**2)*((temperature/288.15)**1.75)
    # DmolVals = [6.21e-5, 6.54e-5, 6.95e-5]
    # Dmol = 1000
    # if temperature == 288.15:
    #     Dmol = DmolVals[0]
    # elif temperature == 298.15:
    #     Dmol = DmolVals[1]
    # elif temperature == 308.15:
    #     Dmol = DmolVals[2] 
    # else:
    #     Dmol = 100

    # for jj in range(n):
    #     Dmaceff[jj] = Dmac / (epsilon + computedqbydc([c[jj]],[q[jj]], temperature, isothermModel, adsorbentDensity) * (1 - epsilon))
    #     # Dmaceff[jj] = np.min([Dmaceff[jj],Dmol])
    #     kmiceff[jj] = kmic / computedlnqbydlnp([c[jj]],[q[jj]], temperature, isothermModel, adsorbentDensity)

    # Dmaceff[:] =  Dmac / (epsilon + computedqbydc([np.mean(c[0:-2])],[np.mean(q[0:-2])], temperature, isothermModel, adsorbentDensity) * (1 - epsilon))
    Dmaceff[:] =  Dmac 
    kmiceff[:] =  kmic
    
    # Dmaceff[-1] = Dmaceff[-2]*10000
    D2cDx2[0] = 6 * Dmaceff[0] / (epsilon * deltar ** 2) * (c[1] - c[0]) 
    heatAds = heatofAdsorption(c, temperature, isothermModel, adsorbentDensity)
    # pdb.set_trace()
    
    for i in range(1, n-2):   
        D2cDx2[i] = (Dmaceff[i] / (epsilon * 2 * (i) * deltar ** 2)) * ((i + 2) * c[i + 1] - 2 * (i) * c[i] + (i - 2) * c[i - 1])
    
    # pdb.set_trace()
    for i in range(0, n-2): 
        # if i == 0:
        DqDt[i] = kmiceff[i] * (computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity) - q[i])
        DcDt[i] = D2cDx2[i] - ((1 - epsilon) / epsilon) * DqDt[i]
            # pdb.set_trace()
        # else:
        #     DqDt[i] = kmiceff[i] * (computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity) - q[i])
        #     DcDt[i] = D2cDx2[i] - ((1 - epsilon) / epsilon) * 3 / ( ((i+1)*deltar)**3 ) * np.trapz(DqDt[0:i] * r[0:i] ** 2, r[0:i])
    
    # volMix = volGas
    volMix = (volGas+volSorbent)
    # volMix = 0.57e-6
    DnDt = (volSorbent*3 / ( (Rp)**3 ) * np.trapz(DqDt[0:n-2] * r[0:-2] ** 2, r[0:-2])+
            volGas*3 /     ( (Rp)**3 ) * np.trapz(DcDt[0:n-2] * r[0:-2] ** 2, r[0:-2]))/(volSorbent+volGas)
    flowOut = volFlow - ((volSorbent+volGas)*(Rg*temperature)/Ptotal)*DnDt;
    DyDt = 1/(volMix) * ((volFlow*0 - flowOut*c[-2]*(Rg*temperature)/Ptotal) - ((volSorbent+volGas)*(Rg*temperature)/Ptotal)*DnDt);
    DcDt[-2] = DyDt*Ptotal/(Rg*temperature)
    DcDt[-1] = 0
    DqDt[-2:] = 0
    # pdb.set_trace()

    DfDt = np.concatenate([DcDt, DqDt])
    # pdb.set_trace()
    return DfDt

def DiffusionAdsorption1Dtau(Y0, tspan, volSorbent, volGas, adsorbentDensity, epsilon, volFlow, temperature, Ptotal, isothermModelAll, rateConstant_1, rateConstant_2, rateConstant_3, rpore, Dpvals):
    import numpy as np
    from scipy.integrate import odeint
    # import pdb
    # import matplotlib.pyplot as plt
    # from numpy import load
    # import os
    # import matplotlib.pyplot as plt
    # import auxiliaryFunctions
    import scipy
    # import nbkode
    # plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    # from scipy.integrate import odeint, solve_ivp, ode

    # Constants and others
    Rg = 8.314  # Universal gas constant [J/molK]
    # pdb.set_trace()
    Rp = ((volSorbent + volGas) / (4/3 * np.pi))**(1/3)  # pellet radius [m]]
    n = 100
    r = np.linspace(0,Rp+2*Rp/n, n)  # discretize radial domain
    
    # pdb.set_trace()
    # Simulate model for input isotherms
    for isothermModel in isothermModelAll:
        # Initial conditions
        c0 = np.zeros(n)  # concentration in macropore [mol/m^3]
        c0[:] = Ptotal * Y0 / (Rg * temperature)  # Initial condition c(r,0) = c0 [mol/m^3]
        c0[-1] = 0
        q0 = computeEquilibrium(c0, temperature, isothermModel, adsorbentDensity)  # Initial condition equilibrium q(r,0) = q*(c,T) [mol/m^3]
        q0[-2:] = 0
        x0 = np.concatenate([c0, q0])    
        inputArgs = r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp, rpore, Dpvals

        # ODE solver
        scipy.integrate.ode(radialDiffusionAdsorption1Dtau).set_integrator('vode', method='bdf', order=15)
        # scipy.integrate.ode(radialDiffusionAdsorption1D).set_integrator('lsoda',first_step =0.001)
        Y = odeint(radialDiffusionAdsorption1Dtau, x0, tspan, args=inputArgs)
        
        
        # solver = nbkode.BDF6(radialDiffusionAdsorption1D, 0, x0, params = inputArgs)
        
        # tspan, Y = solver.run(tspan)
        
        moleGas = np.zeros(len(tspan))
        moleSolid = np.zeros(len(tspan))
        qAverage_i = np.zeros(len(tspan))
        for jj in range(len(tspan)):
            qAverage_i[jj] = 3 / (Rp**3) *             np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
            moleGas[jj] = volGas * 3 / (Rp**3) *       np.trapz(Y[jj, 0:n-2]   * r[0:-2] ** 2, r[0:-2])
            moleSolid[jj] = volSorbent * 3 / (Rp**3) * np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
        moleTotal = moleSolid + moleGas
        moleRate = np.gradient(moleTotal, tspan)
        volRate = -moleRate * Rg * temperature / Ptotal
        yOut = Y[:,n-2]*(Rg*temperature)/Ptotal
        volFlowOut = (volRate + volFlow)
        qAverage = qAverage_i
        yOut[yOut < 1e-5] = 1e-5
        volFlowOut[volFlowOut < volFlow] = volFlow
        volFlowOut[volFlowOut > 100*volFlow] = 100*volFlow
        qAverage[qAverage < 1e-6] = 1e-6
        # pdb.set_trace()
    
    return tspan, Y, r, yOut, volFlowOut, qAverage

def radialDiffusionAdsorption1Dtau(x, t, r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp, rpore, Dpvals):
    import numpy as np
    # import computedqbydc
    # import computedlnqbydlnp
    # import computeEquilibrium
    import pdb
    Rg = 8.314
    Ptotal = 1e5
    DcDt = np.zeros(n)
    DqDt = np.zeros(n)
    DyDt = np.zeros(n)

    deltar = r[1] - r[0]
    D2cDx2 = np.zeros(n)
    
    c = x[0:n]
    q = x[n:2*n]
    
    
    # if t == 0:
    #     c[-2] = Y0*Ptotal/(Rg*temperature)
    
    Dmaceff = np.zeros(n)
    kmiceff = np.zeros(n)
    kmic = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))
    # Dmac = rateConstant_3*(Rp**2)*(temperature**0.5)
    DmolVals = [5.62e-5, 5.95e-5, 6.29e-5]
    # rpore = 107e-9
    Dk = (2/3 * rpore * (8*8.314*temperature/(np.pi*0.044))**0.5)

    if temperature == 288.15:
        Dmol = DmolVals[0]
        Dp = Dpvals[0]
    elif temperature == 298.15:
        Dmol = DmolVals[1]
        Dp = Dpvals[1]
    elif temperature == 308.15:
        Dmol = DmolVals[2] 
        Dp = Dpvals[2]
    else:
        Dmol = 0

    # for jj in range(n):
    #     Dmaceff[jj] = Dmac / (epsilon + computedqbydc([c[jj]],[q[jj]], temperature, isothermModel, adsorbentDensity) * (1 - epsilon))
    #     # Dmaceff[jj] = np.min([Dmaceff[jj],Dmol])
    #     kmiceff[jj] = kmic / computedlnqbydlnp([c[jj]],[q[jj]], temperature, isothermModel, adsorbentDensity)

    # Dmaceff[:] =  Dmac / (epsilon + computedqbydc([np.mean(c[0:-2])],[np.mean(q[0:-2])], temperature, isothermModel, adsorbentDensity) * (1 - epsilon))
    # for jj in range(n):
    #     fac = 1-c[jj]*Rg*temperature/1e5*(1-(44/4)**0.5)
    #     Dmaceff[jj] = epsilon/rateConstant_3*(1/((1/Dk+fac/Dmol)))
    # Dmaceff[:] =  epsilon/rateConstant_3*(1/((1/Dk+1/Dmol))) 
    Dmaceff[:] =  epsilon/rateConstant_3*(Dp) 
    kmiceff[:] =  kmic
    
    # Dmaceff[-1] = Dmaceff[-2]*10000
    D2cDx2[0] = 6 * Dmaceff[0] / (epsilon * deltar ** 2) * (c[1] - c[0]) 
    heatAds = heatofAdsorption(c, temperature, isothermModel, adsorbentDensity)
    # pdb.set_trace()

    # for i in range(1, n-1):   
    #     D2cDx2[i] = (Dmaceff[i] / (epsilon * 2 * (i) * deltar ** 2)) * ((i + 2) * c[i + 1] - 2 * (i) * c[i] + (i - 2) * c[i - 1]) + \
    #                 (Dmaceff[i + 1] / (epsilon * 2 * deltar ** 2))   * (c[i + 1] - c[i]) + \
    #                 (Dmaceff[i - 1] / (epsilon * 2 * deltar ** 2))   * (c[i - 1] - c[i])
    
    # for i in range(1, n-1):   
    #     D2cDx2[i] = (Dmaceff[i] / (epsilon * (i) * deltar ** 2)) * ((i + 1) * c[i + 1] - 2 * (i) * c[i] + (i - 1) * c[i - 1]) 
    
    for i in range(1, n-1):   
        D2cDx2[i] = (Dmaceff[i] / (epsilon * 2 * (i) * deltar ** 2)) * ((i + 2) * c[i + 1] - 2 * (i) * c[i] + (i - 2) * c[i - 1])

    # pdb.set_trace()
    for i in range(0, n-2): 
        # if i == 0:
        DqDt[i] = kmiceff[i] * (computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity) - q[i])
        DcDt[i] = D2cDx2[i] - ((1 - epsilon) / epsilon) * DqDt[i]


    volMix = (volGas+volSorbent)
    # volMix = 0.0001*(volGas+volSorbent)
    # volMix = 0.57e-6
    # volMix = 2.54e-6
    DnDt = (volSorbent*3 / ( (Rp)**3 ) * np.trapz(DqDt[0:n-1] * r[0:-1] ** 2, r[0:-1])+
            volGas*3 /     ( (Rp)**3 ) * np.trapz(DcDt[0:n-1] * r[0:-1] ** 2, r[0:-1]))/(volSorbent+volGas)
    flowOut = volFlow - ((volSorbent+volGas)*(Rg*temperature)/Ptotal)*DnDt
    DyDt = 1/(volMix) * ((volFlow*0 - flowOut*c[-2]*(Rg*temperature)/Ptotal) - ((volSorbent+volGas)*(Rg*temperature)/Ptotal)*DnDt)
    
    
    # flowOut = volFlow - ((volSorbent+volGas)*(Rg*temperature)/Ptotal)*DcDt[-2];
    # DyDt = 1/(volMix) * ((volFlow*0 - flowOut*c[-2]*(Rg*temperature)/Ptotal) - ((volSorbent+volGas)*(Rg*temperature)/Ptotal)*DcDt[-2]);
    DcDt[-2] = DyDt*Ptotal/(Rg*temperature)
    DcDt[-1] = 0
    DqDt[-2:] = 0
    # pdb.set_trace()

    DfDt = np.concatenate([DcDt, DqDt])
    # pdb.set_trace()
    return DfDt

def DiffusionAdsorption1DNItau(Y0, tspan, volSorbent, volGas, adsorbentDensity, epsilon, volFlow, temperature, Ptotal, isothermModelAll, rateConstant_1, rateConstant_2, rateConstant_3, rpore):
    import numpy as np
    from scipy.integrate import odeint, solve_ivp, ode
    import pdb
    # import matplotlib.pyplot as plt
    # from numpy import load
    # import os
    # import matplotlib.pyplot as plt
    # import auxiliaryFunctions
    import scipy
    # import nbkode
    # plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    # from scipy.integrate import odeint, solve_ivp, ode

    # Constants and others
    Rg = 8.314  # Universal gas constant [J/molK]
    # pdb.set_trace()
    Rp = ((volSorbent + volGas) / (4/3 * np.pi))**(1/3)  # pellet radius [m]]
    n = 50
    r = np.linspace(0,Rp+2*Rp/n, n)  # discretize radial domain
    
    # pdb.set_trace()
    # Simulate model for input isotherms
    for isothermModel in isothermModelAll:
        # Initial conditions
        # timeInt = (0.0,max(tspan))
        c0 = np.zeros(n)  # concentration in macropore [mol/m^3]
        c0[:] = Ptotal * Y0 / (Rg * temperature)  # Initial condition c(r,0) = c0 [mol/m^3]
        T0 = np.zeros(n)
        T0[:] = temperature  # Initial condition c(r,0) = c0 [mol/m^3]
        c0[-1] = 0
        q0 = computeEquilibrium(c0, temperature, isothermModel, adsorbentDensity)  # Initial condition equilibrium q(r,0) = q*(c,T) [mol/m^3]
        q0[-2:] = 0
        x0 = np.concatenate([c0, q0, T0])    
        inputArgs = r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp, rpore

        # ODE solver
        # scipy.integrate.ode(radialDiffusionAdsorption1DNItau).set_integrator('vode', method='bdf', order=15)
        # scipy.integrate.ode(radialDiffusionAdsorption1DNItau).set_integrator('lsoda',first_step =0.001)
        Y = odeint(radialDiffusionAdsorption1DNItau, x0, tspan, args=inputArgs,rtol=1e-6, atol=1e-6, mxordn=15)
        # Y = solve_ivp(radialDiffusionAdsorption1DNItau, timeInt, x0, 
        #                       method='Radau', t_eval = tspan, args = inputArgs)
        # tspan, Y = solver.run(tspan)
        
        moleGas = np.zeros(len(tspan))
        moleSolid = np.zeros(len(tspan))
        qAverage_i = np.zeros(len(tspan))
        tempVals = Y[:,2*n:3*n]
        Tvals = tempVals[:,-1]
        for jj in range(len(tspan)):
            qAverage_i[jj] = 3 / (Rp**3) *             np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
            moleGas[jj] = volGas * 3 / (Rp**3) *       np.trapz(Y[jj, 0:n-2]   * r[0:-2] ** 2, r[0:-2])
            moleSolid[jj] = volSorbent * 3 / (Rp**3) * np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
        moleTotal = moleSolid + moleGas
        moleRate = np.gradient(moleTotal, tspan)
        volRate = -moleRate * Rg * Tvals / Ptotal
        yOut = Y[:,n-2]*(Rg*Tvals)/Ptotal
        volFlowOut = (volRate + volFlow)
        qAverage = qAverage_i
        yOut[yOut < 1e-5] = 1e-5
        volFlowOut[volFlowOut < volFlow] = volFlow
        volFlowOut[volFlowOut > 100*volFlow] = 100*volFlow
        qAverage[qAverage < 1e-6] = 1e-6
        
        # pdb.set_trace()
    
    return tspan, Y, r, yOut, volFlowOut, qAverage, tempVals

def radialDiffusionAdsorption1DNItau(x, t, r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp, rpore):
    import numpy as np
    # import computedqbydc
    # import computedlnqbydlnp
    # import computeEquilibrium
    import pdb
    Rg = 8.314
    Ptotal = 1e5
    DcDt = np.zeros(n)
    DqDt = np.zeros(n)
    DTDt = np.zeros(n)

    deltar = r[1] - r[0]
    D2cDx2 = np.zeros(n)
    # pdb.set_trace()
    c = x[0:n]
    q = x[n:2*n]
    T = x[2*n:3*n]
    
    # if t == 0:
    #     c[-2] = Y0*Ptotal/(Rg*temperature)
    
    Dmaceff = np.zeros(n)
    kmiceff = np.zeros(n)
    kmic = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*temperature))
    # Dmac = rateConstant_3*(Rp**2)*(temperature**0.5)
    DmolVals = [5.62e-5, 5.95e-5, 6.29e-5]
    # rpore = 107e-9
    Dk = (2/3 * rpore * (8*8.314*T[-1]/(np.pi*0.044))**0.5)

    if temperature == 288.15:
        Dmol = DmolVals[0]
    elif temperature == 298.15:
        Dmol = DmolVals[1]
    elif temperature == 308.15:
        Dmol = DmolVals[2] 
    else:
        Dmol = 0

    Dmol = (0.0033*T[-1]   -0.4035)*1e-4
    # for jj in range(n):
    #     Dmaceff[jj] = Dmac / (epsilon + computedqbydc([c[jj]],[q[jj]], temperature, isothermModel, adsorbentDensity) * (1 - epsilon))
    #     # Dmaceff[jj] = np.min([Dmaceff[jj],Dmol])
    #     kmiceff[jj] = kmic / computedlnqbydlnp([c[jj]],[q[jj]], temperature, isothermModel, adsorbentDensity)

    # Dmaceff[:] =  Dmac / (epsilon + computedqbydc([np.mean(c[0:-2])],[np.mean(q[0:-2])], temperature, isothermModel, adsorbentDensity) * (1 - epsilon))

    Dmac = epsilon/rateConstant_3*(1/((1/Dk+1/Dmol)))
    Dmaceff[:] =  Dmac 
    kmiceff[:] =  kmic
    
    # Dmaceff[-1] = Dmaceff[-2]*10000
    D2cDx2[0] = 6 * Dmaceff[0] / (epsilon * deltar ** 2) * (c[1] - c[0]) 
    heatAds = heatofAdsorption(c, T[-1], isothermModel, adsorbentDensity)
    # heatAds[:] =     isothermModel[2]
    Cs =  1344*adsorbentDensity;
    a = 3/Rp
    conductivity = 0.15
    kinVis = 11.69e-5
    crossArea = (np.pi*0.008**2)/4
    Diameter = 2*Rp
    U = volFlow/(crossArea-np.pi*Rp**2)
    Re = U*Diameter/kinVis
    Pr = 0.67
    Nu = 2 + (0.4*Re**0.5 + 0.06*Re**0.4)*Pr**0.4
    # Nu = 2 + 0.6*Re**0.5*Pr**(1/3)
    ha = conductivity*Nu/(2*Rp)*a
    for i in range(1, n-2):   
        # D2cDx2[i] = (Dmaceff[i] / (epsilon * 2 * (i) * deltar ** 2)) * ((i + 2) * c[i + 1] - 2 * (i) * c[i] + (i - 2) * c[i - 1]) + \
        #             (Dmaceff[i + 1] / (epsilon * 2 * deltar ** 2))   * (c[i + 1] - c[i]) + \
        #             (Dmaceff[i - 1] / (epsilon * 2 * deltar ** 2))   * (c[i - 1] - c[i])
        D2cDx2[i] = (Dmaceff[i] / (epsilon * (i) * deltar ** 2)) * ((i + 1) * c[i + 1] - 2 * (i) * c[i] + (i - 1) * c[i - 1])
    
    # pdb.set_trace()
    for i in range(0, n-2): 
        # if i == 0:
        DqDt[i] = kmiceff[i] * (computeEquilibrium(c[i], T[-1], isothermModel, adsorbentDensity) - q[i])
        DcDt[i] = D2cDx2[i] - ((1 - epsilon) / epsilon) * DqDt[i]
            # pdb.set_trace()
        # else:
        #     DqDt[i] = kmiceff[i] * (computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity) - q[i])
        #     DcDt[i] = D2cDx2[i] - ((1 - epsilon) / epsilon) * 3 / ( ((i+1)*deltar)**3 ) * np.trapz(DqDt[0:i] * r[0:i] ** 2, r[0:i])
    # pdb.set_trace()
    # volMix = volGas
    volMix = (volGas+volSorbent)
    # volMix = 0.57e-6
    DnDt = (volSorbent*3 / ( (Rp)**3 ) * np.trapz(DqDt[0:n-2] * r[0:-2] ** 2, r[0:-2])+
            volGas*3 /     ( (Rp)**3 ) * np.trapz(DcDt[0:n-2] * r[0:-2] ** 2, r[0:-2]))/(volSorbent+volGas)
    flowOut = volFlow - ((volSorbent+volGas)*(Rg*T[-1])/Ptotal)*DnDt;
    DyDt = 1/(volMix) * ((volFlow*0 - flowOut*c[-2]*(Rg*T[-1])/Ptotal) - ((volSorbent+volGas)*(Rg*T[-1])/Ptotal)*DnDt);
    DcDt[-2] = DyDt*Ptotal/(Rg*T[-1])
    DcDt[-1] = 0
    DqDt[-2:] = 0
    # pdb.set_trace()
    DTDt[:] = 1/Cs *(DnDt*heatAds[0]+(ha)*(temperature-T[1]))

    DfDt = np.concatenate([DcDt, DqDt, DTDt])
    # pdb.set_trace()
    return DfDt


def DiffusionAdsorption1DNI(Y0, tspan, volSorbent, volGas, adsorbentDensity, epsilon, volFlow, temperature, Ptotal, isothermModelAll, rateConstant_1, rateConstant_2, rateConstant_3):
    import numpy as np
    from scipy.integrate import odeint
    # import pdb
    # import matplotlib.pyplot as plt
    # from numpy import load
    # import os
    # import matplotlib.pyplot as plt
    # import auxiliaryFunctions
    import scipy
    # import nbkode
    # plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    # from scipy.integrate import odeint, solve_ivp, ode

    # Constants and others
    Rg = 8.314  # Universal gas constant [J/molK]
    # pdb.set_trace()
    Rp = ((volSorbent + volGas) / (4/3 * np.pi))**(1/3)  # pellet radius [m]]
    n = 60
    r = np.linspace(0,Rp+2*Rp/n, n)  # discretize radial domain
    
    # pdb.set_trace()
    # Simulate model for input isotherms
    for isothermModel in isothermModelAll:
        # Initial conditions
        c0 = np.zeros(n)  # concentration in macropore [mol/m^3]
        c0[:] = Ptotal * Y0 / (Rg * temperature)  # Initial condition c(r,0) = c0 [mol/m^3]
        T0 = np.zeros(n)
        T0[:] = temperature  # Initial condition c(r,0) = c0 [mol/m^3]
        c0[-1] = 0
        q0 = computeEquilibrium(c0, temperature, isothermModel, adsorbentDensity)  # Initial condition equilibrium q(r,0) = q*(c,T) [mol/m^3]
        q0[-2:] = 0
        x0 = np.concatenate([c0, q0, T0])    
        inputArgs = r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp

        # ODE solver
        scipy.integrate.ode(radialDiffusionAdsorption1DNI).set_integrator('vode', method='bdf', order=15)
        # scipy.integrate.ode(radialDiffusionAdsorption1D).set_integrator('lsoda',first_step =0.001)
        Y = odeint(radialDiffusionAdsorption1DNI, x0, tspan, args=inputArgs)
        
        
        # solver = nbkode.BDF6(radialDiffusionAdsorption1D, 0, x0, params = inputArgs)
        
        # tspan, Y = solver.run(tspan)
        
        moleGas = np.zeros(len(tspan))
        moleSolid = np.zeros(len(tspan))
        qAverage_i = np.zeros(len(tspan))
        for jj in range(len(tspan)):
            qAverage_i[jj] = 3 / (Rp**3) *             np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
            moleGas[jj] = volGas * 3 / (Rp**3) *       np.trapz(Y[jj, 0:n-2]   * r[0:-2] ** 2, r[0:-2])
            moleSolid[jj] = volSorbent * 3 / (Rp**3) * np.trapz(Y[jj, n:2*n-2] * r[0:-2] ** 2, r[0:-2])
        moleTotal = moleSolid + moleGas
        moleRate = np.gradient(moleTotal, tspan)
        volRate = -moleRate * Rg * temperature / Ptotal
        yOut = Y[:,n-2]*(Rg*temperature)/Ptotal
        volFlowOut = (volRate + volFlow)
        qAverage = qAverage_i
        yOut[yOut < 1e-5] = 1e-5
        volFlowOut[volFlowOut < volFlow] = volFlow
        volFlowOut[volFlowOut > 100*volFlow] = 100*volFlow
        qAverage[qAverage < 1e-6] = 1e-6
        tempVals = Y[:,2*n:3*n];
        # pdb.set_trace()
    
    return tspan, Y, r, yOut, volFlowOut, qAverage, tempVals

def radialDiffusionAdsorption1DNI(x, t, r, n, isothermModel, temperature, rateConstant_1, rateConstant_2, rateConstant_3, epsilon, adsorbentDensity,volGas, volSorbent, volFlow, Y0, Rp):
    import numpy as np
    # import computedqbydc
    # import computedlnqbydlnp
    # import computeEquilibrium
    import pdb
    Rg = 8.314
    Ptotal = 1e5
    DcDt = np.zeros(n)
    DqDt = np.zeros(n)
    DTDt = np.zeros(n)

    deltar = r[1] - r[0]
    D2cDx2 = np.zeros(n)
    
    c = x[0:n]
    q = x[n:2*n]
    T = x[2*n:3*n]
    
    # if t == 0:
    #     c[-2] = Y0*Ptotal/(Rg*temperature)
    
    Dmaceff = np.zeros(n)
    kmiceff = np.zeros(n)
    kmic = rateConstant_1*np.exp(-rateConstant_2*1000/(Rg*T[-1]))
    # Dmac = rateConstant_3*(Rp**2)*(temperature**0.5)
    Dmac = rateConstant_3*(Rp**2)*((T[-1]/288.15)**1.75)
    # Dmac = rateConstant_3*(Rp**2)*((T[-1]/288.15)**2)

    Dmaceff[:] =  Dmac 
    kmiceff[:] =  kmic
    
    # Dmaceff[-1] = Dmaceff[-2]*10000
    D2cDx2[0] = 6 * Dmaceff[0] / (epsilon * deltar ** 2) * (c[1] - c[0]) 
    heatAds = heatofAdsorption(c, T[-1], isothermModel, adsorbentDensity)
    # heatAds[:] =     isothermModel[2]
    Cs =  1344*adsorbentDensity;
    a = 3/Rp
    conductivity = 0.15
    kinVis = 11.69e-5
    crossArea = (np.pi*0.008**2)/4
    Diameter = 2*Rp
    U = volFlow/(crossArea-np.pi*Rp**2)
    Re = U*Diameter/kinVis
    Pr = 0.67
    Nu = 2 + (0.4*Re**0.5 + 0.06*Re**0.4)*Pr**0.4
    # Nu = 2 + 0.6*Re**0.5*Pr**(1/3)
    ha = conductivity*Nu/(2*Rp)*a
    for i in range(1, n-2):   
        D2cDx2[i] = (Dmaceff[i] / (epsilon * 2 * (i) * deltar ** 2)) * ((i + 2) * c[i + 1] - 2 * (i) * c[i] + (i - 2) * c[i - 1]) + \
                    (Dmaceff[i + 1] / (epsilon * 2 * deltar ** 2))   * (c[i + 1] - c[i]) + \
                    (Dmaceff[i - 1] / (epsilon * 2 * deltar ** 2))   * (c[i - 1] - c[i])
    
    # pdb.set_trace()
    for i in range(0, n-2): 
        # if i == 0:
        DqDt[i] = kmiceff[i] * (computeEquilibrium(c[i], T[-1], isothermModel, adsorbentDensity) - q[i])
        DcDt[i] = D2cDx2[i] - ((1 - epsilon) / epsilon) * DqDt[i]
            # pdb.set_trace()
        # else:
        #     DqDt[i] = kmiceff[i] * (computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity) - q[i])
        #     DcDt[i] = D2cDx2[i] - ((1 - epsilon) / epsilon) * 3 / ( ((i+1)*deltar)**3 ) * np.trapz(DqDt[0:i] * r[0:i] ** 2, r[0:i])
    # pdb.set_trace()

    # volMix = volGas
    volMix = (volGas+volSorbent)
    # volMix = 0.57e-6
    DnDt = (volSorbent*3 / ( (Rp)**3 ) * np.trapz(DqDt[0:n-2] * r[0:-2] ** 2, r[0:-2])+
            volGas*3 /     ( (Rp)**3 ) * np.trapz(DcDt[0:n-2] * r[0:-2] ** 2, r[0:-2]))/(volSorbent+volGas)
    flowOut = volFlow - ((volSorbent+volGas)*(Rg*T[-1])/Ptotal)*DnDt;
    DyDt = 1/(volMix) * ((volFlow*0 - flowOut*c[-2]*(Rg*T[-1])/Ptotal) - ((volSorbent+volGas)*(Rg*T[-1])/Ptotal)*DnDt);
    DcDt[-2] = DyDt*Ptotal/(Rg*T[-1])
    DcDt[-1] = 0
    DqDt[-2:] = 0
    # pdb.set_trace()
    DTDt[:] = 1/Cs *(DnDt*heatAds[0]+(ha)*(temperature-T[1]))

    DfDt = np.concatenate([DcDt, DqDt, DTDt])
    # pdb.set_trace()
    return DfDt


def computeEquilibrium(c, temperature, isothermModel, adsorbentDensity):
    import numpy as np
    # import pdb
    
    Rg = 8.314
    # pdb.set_trace()
    isoAffinityA = isothermModel[1] * np.exp(isothermModel[2] / (Rg * temperature))
    isoNumeratorA = isothermModel[0] * isoAffinityA * c
    isoDenominatorA = 1 + isoAffinityA * c

    isoAffinityB = isothermModel[4] * np.exp(isothermModel[5] / (Rg * temperature))
    isoNumeratorB = isothermModel[3] * isoAffinityB * c
    isoDenominatorB = 1 + isoAffinityB * c

    q = adsorbentDensity * (isoNumeratorA / isoDenominatorA + isoNumeratorB / isoDenominatorB)
    return q

def computedlnqbydlnp(c, q, temperature, isothermModel, adsorbentDensity):
    import numpy as np
    # import computeEquilibrium
    # import pdb 
    Rg = 8.314
    dlnqbydlnp = np.zeros(len(c))
    for i in range(len(c)):
        delp = 1
        partialPressure = c[i] * Rg * temperature
        partialPressureUp = partialPressure + delp
        if partialPressure == 0:
            dlnqbydlnp[i] = 1
        elif partialPressure < 0:
            dlnqbydlnp[i] = 1
        else:
            cUp = c[i] + delp / (Rg * temperature)
            
            equilibriumLoading = computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity)
            equilibriumLoadingUp = computeEquilibrium(cUp, temperature, isothermModel, adsorbentDensity)
            
            dellogp = np.log(partialPressureUp) - np.log(partialPressure)
            dellogq = np.log(equilibriumLoadingUp) - np.log(equilibriumLoading)
            
            dlnqbydlnp[i] = dellogq / dellogp
    # pdb.set_trace()
    return dlnqbydlnp


def computedqbydc(c, q, temperature, isothermModel, adsorbentDensity):
    import numpy as np
    # import computeEquilibrium
    Rg = 8.314
    delc = 0.0001
    dqbydc = np.zeros(len(c))
    for i in range(len(c)):
        cUp = c[i] + delc
        if c[i] == 0:
            dqbydc[i] = adsorbentDensity*(isothermModel[0] * isothermModel[1] * np.exp(isothermModel[2] / (Rg * temperature)) + isothermModel[3] * isothermModel[4] * np.exp(isothermModel[5] / (Rg * temperature)))
        elif c[i] < 0:
            dqbydc[i] = adsorbentDensity*(isothermModel[0] * isothermModel[1] * np.exp(isothermModel[2] / (Rg * temperature)) + isothermModel[3] * isothermModel[4] * np.exp(isothermModel[5] / (Rg * temperature)))
        else:
            equilibriumLoading = computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity)
            equilibriumLoadingUp = computeEquilibrium(cUp, temperature, isothermModel, adsorbentDensity)
            
            delc = cUp - c[i]
            delq = equilibriumLoadingUp - equilibriumLoading
            
            dqbydc[i] = delq / delc
            
    return dqbydc

def heatofAdsorption(c, temperature, isothermModel, adsorbentDensity):
    import numpy as np
    # import pdb
    
    Rg = 8.314
    # pdb.set_trace()
    isoAffinityA = isothermModel[1] * np.exp(isothermModel[2] / (Rg * temperature))
    isoNumeratorA1 = isothermModel[0] * isoAffinityA * isothermModel[2]
    isoNumeratorA2 = isothermModel[0] * isoAffinityA
    isoDenominatorA = (1 + isoAffinityA * c)**2

    isoAffinityB = isothermModel[4] * np.exp(isothermModel[5] / (Rg * temperature))
    isoNumeratorB1 = isothermModel[3] * isoAffinityB * isothermModel[5]
    isoNumeratorB2 = isothermModel[3] * isoAffinityB
    isoDenominatorB = (1 + isoAffinityB * c)**2
    
    isonumeratorTotal = isoNumeratorA1/isoDenominatorA + isoNumeratorB1/isoDenominatorB
    isoDenominatorTotal = isoNumeratorA2/isoDenominatorA + isoNumeratorB2/isoDenominatorB

    delH = isonumeratorTotal/isoDenominatorTotal
    return delH



# def computedlnqbydlnp(c, q, temperature, isothermModel, adsorbentDensity):
#     import numpy as np
#     # import computeEquilibrium
#     # import pdb 
#     Rg = 8.314
#     dlnqbydlnp = np.zeros(len(q))
#     for i in range(len(q)):
#         cvals = np.linspace(0,50,1000)
       
#         errorvals = (computeEquilibrium(cvals,temperature,isothermModel,adsorbentDensity)-q[i])**2
#         ind = np.argmin(errorvals)
#         cinit = cvals[ind]
        
#         delp = 1
#         partialPressure = cinit * Rg * temperature
        
#         if partialPressure == 0:
#             dlnqbydlnp[i] = 1
#         elif partialPressure < 0:
#             dlnqbydlnp[i] = 1
#         else:
#             partialPressureUp = partialPressure + delp
#             cUp = cinit + delp / (Rg * temperature)
#             equilibriumLoading = computeEquilibrium(cinit, temperature, isothermModel, adsorbentDensity)
#             equilibriumLoadingUp = computeEquilibrium(cUp, temperature, isothermModel, adsorbentDensity)
            
#             dellogp = np.log(partialPressureUp) - np.log(partialPressure)
#             dellogq = np.log(equilibriumLoadingUp) - np.log(equilibriumLoading)
            
#             dlnqbydlnp[i] = dellogq / dellogp

#     return dlnqbydlnp


# def computedqbydc(c, q, temperature, isothermModel, adsorbentDensity):
#     import numpy as np
#     import pdb
#     Rg = 8.134
#     delc = 0.01
#     dqbydc = np.zeros(len(c))
#     for i in range(len(c)):
#         # cUp = c[i] + delc
#         # if c[i] == 0:
#         #     dqbydc[i] = 1000
#         # elif c[i] < 0:
#         #     dqbydc[i] = 1000
#         if c[i] < 0:
#             c[i] = 0
    
#         # equilibriumLoading = computeEquilibrium(c[i], temperature, isothermModel, adsorbentDensity)
#         # equilibriumLoadingUp = computeEquilibrium(cUp, temperature, isothermModel, adsorbentDensity)
        
#         # delc = cUp - c[i]
#         # delq = equilibriumLoadingUp - equilibriumLoading
        
#         # dqbydc[i] = delq / delc
        
#         cvals = np.linspace(0,60,1000)
       
#         errorvals = (computeEquilibrium(cvals,temperature,isothermModel,adsorbentDensity)-q[i])**2
#         ind = np.argmin(errorvals)
#         cinit = cvals[ind]
        
#         if cinit == 0:
#             cinit = 0.001
#         elif cinit < 0:
#             cinit = 0.001
        
#         cUp = cinit+delc
        
#         equilibriumLoading = computeEquilibrium(cinit, temperature, isothermModel, adsorbentDensity)
#         equilibriumLoadingUp = computeEquilibrium(cUp, temperature, isothermModel, adsorbentDensity)
        
#         delq = equilibriumLoadingUp - equilibriumLoading
        
#         dqbydc[i] = delq / delc
#         # pdb.set_trace()

#         # Rg = 8.314
#         # # pdb.set_trace()
#         # isoAffinityA = isothermModel[1] * np.exp(isothermModel[2] / (Rg * temperature))
#         # isoNumeratorA = isothermModel[0] * isoAffinityA * cinit
#         # isoDenominatorA = 1 + isoAffinityA * cinit
    
#         # isoAffinityB = isothermModel[4] * np.exp(isothermModel[5] / (Rg * temperature))
#         # isoNumeratorB = isothermModel[3] * isoAffinityB * cinit
#         # isoDenominatorB = 1 + isoAffinityB * cinit
        
#         # qA = isoNumeratorA/isoDenominatorA
#         # qB = isoNumeratorB/isoDenominatorB
        
#         # dqbydc[i] = ((1-qA /(isothermModel[0]))**2*isoAffinityA*isothermModel[0]  +   (1-qB /(isothermModel[3]))**2*isoAffinityB*isothermModel[3])*adsorbentDensity
            
#     return dqbydc
