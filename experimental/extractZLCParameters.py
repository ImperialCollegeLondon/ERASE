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
    import multiprocessing # For parallel processing
    import socket
    
    # Change path directory
    # Assumes either running from ERASE or from experimental. Either ways
    # this has to be run from experimental
    if not os.getcwd().split(os.path.sep)[-1] == 'experimental':
        os.chdir("experimental")
    
    # Find out the total number of cores available for parallel processing
    num_cores = multiprocessing.cpu_count()
    
    # Isotherm model type
    modelType = 'SSL'
    
    # Number of times optimization repeated
    numOptRepeat = 10
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Directory of raw data
    mainDir = 'runData'
    # File name of the experiments
    fileName = ['ZLC_ActivatedCarbon_Exp20A_Output.mat',
                'ZLC_ActivatedCarbon_Exp20B_Output.mat',
                'ZLC_ActivatedCarbon_Exp20C_Output.mat']
    
    # NOTE: Dead volume characteristics filename is hardcoded in 
    # simulateCombinedModel. This is because of the python GA function unable
    # to pass arguments
    
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,mainDir,fileName)

    # Define the bounds and the type of the parameters to be optimized   
    # Single-site Langmuir
    if modelType == 'SSL':
        optBounds = np.array(([np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1]))
        optType=np.array(['real','real','real','real'])
        problemDimension = len(optType)
    # Dual-site Langmuir
    elif modelType == 'DSL':
        optBounds = np.array(([np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1]))
        optType=np.array(['real','real','real','real','real','real','real'])
        problemDimension = len(optType)

    # Algorithm parameters for GA
    algorithm_param = {'max_num_iteration':5,
                       'population_size':1600,
                       'mutation_probability':0.1,
                       'crossover_probability': 0.55,
                       'parents_portion': 0.15,
                       'elit_ratio': 0.01,
                       'max_iteration_without_improv':None}
    
    # Minimize an objective function to compute the dead volume and the number of 
    # tanks for the dead volume using GA
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
           fileName = fileName, # Names of file used for fitting
           hostName = socket.gethostname()) # Hostname of the computer) 
        
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
    
    # Reference for the isotherm parameters
    # For SSL isotherm
    if len(x) == 4:
        isoRef = [10, 1e-5, 50e3, 100]
    # For DSL isotherm
    elif len(x) == 7:
        isoRef = [10, 1e-5, 50e3, 10, 1e-5, 50e3, 100]

    # Prepare isotherm model (the first n-1 parameters are for the isotherm model)
    isothermModel = np.multiply(x[0:-1],isoRef[0:-1])

    # Load the names of the file to be used for estimating dead volume characteristics
    filePath = filesToProcess(False,[],[])
    
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

        # Compute the dead volume response using the optimizer parameters
        _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant = x[-1]*isoRef[-1], # Last element is rate constant
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True)

        # Stack mole fraction from experiments and simulation for error 
        # computation
        moleFracExpALL = np.hstack((moleFracExpALL, moleFracExp))
        moleFracSimALL = np.hstack((moleFracSimALL, moleFracSim))

    # Compute the sum of the error for the difference between exp. and sim.
    computedError = computeMLEError(moleFracExpALL,moleFracSimALL)
    return computedError