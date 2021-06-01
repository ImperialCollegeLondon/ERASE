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
# Find the isotherm parameters and the kinetic rate constant by fitting
# the complete response curve from the ZLC experiment. Note that currently
# the isotherm can be SSL or DSL model. The rate constant is assumed to be a
# constant in the LDF model
# Reference: 10.1016/j.ces.2014.12.062
#
# Last modified:
# - 2021-06-01, AK: Add temperature as an input
# - 2021-05-25, AK: Add kinetic mode for estimation
# - 2021-05-24, AK: Improve information passing (for output)
# - 2021-05-13, AK: Change structure to input mass of adsorbent
# - 2021-05-05, AK: Bug fix for MLE error computation
# - 2021-05-05, AK: Modify error computation for dead volume
# - 2021-04-28, AK: Add reference values for isotherm parameters
# - 2021-04-27, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
############################################################################

def extractZLCParameters():
    import numpy as np
    from geneticalgorithm2 import geneticalgorithm2 as ga # GA
    from extractDeadVolume import filesToProcess # File processing script
    import auxiliaryFunctions
    import os
    from numpy import savez
    from numpy import load
    import multiprocessing # For parallel processing
    import socket
    
    # Change path directory
    # Assumes either running from ERASE or from experimental. Either ways
    # this has to be run from experimental
    if not os.getcwd().split(os.path.sep)[-1] == 'experimental':
        os.chdir("experimental")

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Find out the total number of cores available for parallel processing
    num_cores = multiprocessing.cpu_count()

    #####################################
    ###### USER DEFINED PROPERTIES ######     
    
    # Isotherm model type
    modelType = 'SSL'
    
    # Number of times optimization repeated
    numOptRepeat = 10
    
    # Directory of raw data
    mainDir = 'runData'
    # File name of the experiments
    fileName = ['ZLC_ActivatedCarbon_Exp39A_Output.mat',
                'ZLC_ActivatedCarbon_Exp39B_Output.mat',
                'ZLC_ActivatedCarbon_Exp39C_Output.mat',
                'ZLC_ActivatedCarbon_Exp39D_Output.mat',
                'ZLC_ActivatedCarbon_Exp39E_Output.mat',
                'ZLC_ActivatedCarbon_Exp39F_Output.mat']
    
    # Temperature (for each experiment)
    temperature = [313.15, 313.15, 313.15, 313.15, 313.15, 313.15]
    
    # Dead volume model
    deadVolumeFile = 'deadVolumeCharacteristics_20210528_1319_318b280.npz'

    # Isotherm model (if fitting only kinetic constant)
    isothermFile = 'zlcParameters_20210525_1610_a079f4a.npz'

    # Adsorbent properties
    # Adsorbent density [kg/m3]
    # This has to be the skeletal density
    adsorbentDensity = 1950 # Activated carbon skeletal density [kg/m3]
    # Particle porosity
    particleEpsilon = 0.61
    # Particle mass [g]
    massSorbent = 0.0625
    
    # Threshold factor (If -ngative infinity not used, if not need a float)
    # This is used to split the compositions into two distint regions
    thresholdFactor = -np.inf
    
    #####################################
    #####################################
    
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,mainDir,fileName,'ZLC')

    # Define the bounds and the type of the parameters to be optimized   
    # Single-site Langmuir
    if modelType == 'SSL':
        optBounds = np.array(([np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1]))
        optType=np.array(['real','real','real','real'])
        problemDimension = len(optType)
        isoRef = [10, 1e-5, 50e3, 100] # Reference for the isotherm parameters
        isothermFile = [] # Isotherm file is empty as it is fit
        paramIso = [] # Isotherm parameters is empty as it is fit

    # Dual-site Langmuir
    elif modelType == 'DSL':
        optBounds = np.array(([np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1]))
        optType=np.array(['real','real','real','real','real','real','real'])
        problemDimension = len(optType)
        isoRef = [10, 1e-5, 50e3, 10, 1e-5, 50e3, 100] # Reference for the isotherm parameters
        isothermFile = [] # Isotherm file is empty as it is fit
        paramIso = [] # Isotherm parameters is empty as it is fit

    # Kinetic constant only
    elif modelType == 'Kinetic':
        optBounds = np.array([[np.finfo(float).eps,1]])
        optType=np.array(['real'])
        problemDimension = len(optType)
        isoRef = [100] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'simulationResults/'
        modelOutputTemp = load(isothermDir+isothermFile, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"]
        parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        paramIso = np.multiply(modelNonDim,parameterRefTemp)

    # Initialize the parameters used for ZLC fitting process
    fittingParameters(True,temperature,deadVolumeFile,adsorbentDensity,particleEpsilon,
                      massSorbent,isoRef,thresholdFactor,paramIso)

    # Algorithm parameters for GA
    algorithm_param = {'max_num_iteration':5,
                       'population_size':1600,
                       'mutation_probability':0.1,
                       'crossover_probability': 0.55,
                       'parents_portion': 0.15,
                       'elit_ratio': 0.01,
                       'max_iteration_without_improv':None}
    
    # Minimize an objective function to compute the equilibrium and kinetic 
    # parameters from ZLC experiments
    model = ga(function = ZLCObjectiveFunction, dimension=problemDimension, 
                               variable_type_mixed = optType,
                               variable_boundaries = optBounds,
                               algorithm_parameters=algorithm_param)
    
    # Call the GA optimizer using multiple cores
    model.run(set_function=ga.set_function_multiprocess(ZLCObjectiveFunction,
                                                         n_jobs = num_cores),
              no_plot = True)
    # Repeat the optimization with the last generation repeated numOptRepeat
    # times (for better accuracy)
    for ii in range(numOptRepeat):
        model.run(set_function=ga.set_function_multiprocess(ZLCObjectiveFunction,
                                                             n_jobs = num_cores),
                  start_generation=model.output_dict['last_generation'], no_plot = True)
    
    # Save the array concentration into a native numpy file
    # The .npz file is saved in a folder called simulationResults (hardcoded)
    filePrefix = "zlcParameters"
    saveFileName = filePrefix + "_" + currentDT + "_" + gitCommitID;
    savePath = os.path.join('..','simulationResults',saveFileName)
    
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationResults')):
        os.mkdir(os.path.join('..','simulationResults'))
    
    # Save the output into a .npz file
    savez (savePath, modelOutput = model.output_dict, # Model output
           optBounds = optBounds, # Optimizer bounds
           algoParameters = algorithm_param, # Algorithm parameters
           numOptRepeat = numOptRepeat, # Number of times optimization repeated
           fileName = fileName, # Names of file used for fitting
           temperature = temperature, # Temperature [K]
           deadVolumeFile = deadVolumeFile, # Dead volume file used for parameter estimation
           isothermFile = isothermFile, # Isotherm parameters file, if only kinetics estimated
           adsorbentDensity = adsorbentDensity, # Adsorbent density [kg/m3]
           particleEpsilon = particleEpsilon, # Particle voidage [-]
           massSorbent = massSorbent, # Mass of sorbent [g]
           parameterReference = isoRef, # Parameter references [-]
           mleThreshold = thresholdFactor, # Threshold for MLE composition split [-]
           hostName = socket.gethostname()) # Hostname of the computer
    
    # Remove all the .npy files genereated from the .mat
    # Load the names of the file to be used for estimating ZLC parameters
    filePath = filesToProcess(False,[],[],'ZLC')
    # Loop over all available files    
    for ii in range(len(filePath)):
        os.remove(filePath[ii])
    
    # Return the optimized values
    return model.output_dict
    
# func: deadVolObjectiveFunction
# For use with GA, the function accepts only one input (parameters from the 
# optimizer)
def ZLCObjectiveFunction(x):
    import numpy as np
    from numpy import load
    from extractDeadVolume import filesToProcess # File processing script
    from simulateCombinedModel import simulateCombinedModel
    from computeMLEError import computeMLEError

    # Get the zlc parameters needed for the solver
    temperature, deadVolumeFile, adsorbentDensity, particleEpsilon, massSorbent, isoRef, thresholdFactor, paramIso = fittingParameters(False,[],[],[],[],[],[],[],[])

    # Volume of sorbent material [m3]
    volSorbent = (massSorbent/1000)/adsorbentDensity
    # Volume of gas in pores [m3]
    volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

    # Prepare isotherm model (the first n-1 parameters are for the isotherm model)
    if len(paramIso) != 0:
        isothermModel = paramIso[0:-1] # Use this if isotherm parameter provided (for kinetics only)
    else:        
        isothermModel = np.multiply(x[0:-1],isoRef[0:-1]) # Use this if both equilibrium and kinetics is fit

    # Load the names of the file to be used for estimating zlc parameters
    filePath = filesToProcess(False,[],[],'ZLC')
    
    # Parse out number of data points for each experiment (for downsampling)
    numPointsExp = np.zeros(len(filePath))
    for ii in range(len(filePath)): 
        # Load experimental molefraction
        timeElapsedExp = load(filePath[ii])["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
        
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)
    
    # Initialize error for objective function
    computedError = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])    

    # Loop over all available files    
    for ii in range(len(filePath)):
        # Initialize outputs
        moleFracSim = []  
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExpTemp = load(filePath[ii])["timeElapsed"].flatten()
        moleFracExpTemp = load(filePath[ii])["moleFrac"].flatten()
        flowRateTemp = load(filePath[ii])["flowRate"].flatten()
        timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
        moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
        flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))] # [cc/s]
                
        # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
        timeInt = timeElapsedExp

        # Compute the composite response using the optimizer parameters
        _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant = x[-1]*isoRef[-1], # Last element is rate constant
                                                    temperature = temperature[ii], # Temperature [K]
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    deadVolumeFile = str(deadVolumeFile),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas)

        # Stack mole fraction from experiments and simulation for error 
        # computation
        moleFracExpALL = np.hstack((moleFracExpALL, moleFracExp))
        moleFracSimALL = np.hstack((moleFracSimALL, moleFracSim))

    # Compute the sum of the error for the difference between exp. and sim.
    computedError = computeMLEError(moleFracExpALL,moleFracSimALL,thresholdFactor=thresholdFactor)
    return computedError

# func: fittingParameters
# Parses dead volume calibration file, adsorbent density, voidage, mass to 
# be used for parameter estimation, parameter references and threshold for MLE
# This is done because the ga cannot handle additional user inputs
def fittingParameters(initFlag,temperature,deadVolumeFile,adsorbentDensity,particleEpsilon,massSorbent,isoRef, thresholdFactor,paramIso):
    from numpy import savez
    from numpy import load
    # Process the data for python (if needed)
    if initFlag:
        # Save the necessary inputs to a temp file
        dummyFileName = 'tempFittingParametersZLC.npz'
        savez (dummyFileName, temperature = temperature,
               deadVolumeFile = deadVolumeFile,
               adsorbentDensity=adsorbentDensity,
               particleEpsilon=particleEpsilon,
               massSorbent=massSorbent,
               isoRef=isoRef,
               thresholdFactor=thresholdFactor,
               paramIso = paramIso)
    # Returns the path of the .npz file to be used 
    else:
    # Load the dummy file with temperature, deadVolumeFile, adsorbent density, particle voidage,
    # and mass of sorbent
        dummyFileName = 'tempFittingParametersZLC.npz'
        temperature = load (dummyFileName)["temperature"]
        deadVolumeFile = load (dummyFileName)["deadVolumeFile"]
        adsorbentDensity = load (dummyFileName)["adsorbentDensity"]
        particleEpsilon = load (dummyFileName)["particleEpsilon"]
        massSorbent = load (dummyFileName)["massSorbent"]
        isoRef = load (dummyFileName)["isoRef"]
        thresholdFactor = load (dummyFileName)["thresholdFactor"]
        paramIso = load (dummyFileName)["paramIso"]
        return temperature, deadVolumeFile, adsorbentDensity, particleEpsilon, massSorbent, isoRef, thresholdFactor, paramIso