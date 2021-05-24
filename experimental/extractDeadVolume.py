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
# Find the dead volume and the number of tanks to describe the dead volume 
# using the tanks in series (TIS) for the ZLC
# Reference: 10.1016/j.ces.2008.02.023
# The methodolgy is slighlty modified to incorporate diffusive pockets using
# compartment models (see Levenspiel, chapter 12) or Lisa Joss's article
# Reference: 10.1007/s10450-012-9417-z
#
# Last modified:
# - 2021-05-24, AK: Improve information passing (for output)
# - 2021-05-05, AK: Bug fix for MLE error computation
# - 2021-05-05, AK: Bug fix for error computation
# - 2021-05-04, AK: Modify error computation for dead volume
# - 2021-04-27, AK: Cosmetic changes to structure
# - 2021-04-21, AK: Change model to fix split velocity
# - 2021-04-20, AK: Change model to flow dependent split
# - 2021-04-20, AK: Implement time-resolved experimental flow rate for DV
# - 2021-04-15, AK: Modify GA parameters and add penalty function
# - 2021-04-14, AK: Bug fix
# - 2021-04-14, AK: Change strucure and perform series of parallel CSTRs
# - 2021-04-12, AK: Add functionality for multiple experiments
# - 2021-03-25, AK: Estimate parameters using experimental data
# - 2021-03-17, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def extractDeadVolume():
    import numpy as np
    from geneticalgorithm2 import geneticalgorithm2 as ga # GA
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

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()    

    # Find out the total number of cores available for parallel processing
    num_cores = multiprocessing.cpu_count()
    
    #####################################
    ###### USER DEFINED PROPERTIES ###### 
    
    # Number of times optimization repeated
    numOptRepeat = 10

    # Directory of raw data    
    mainDir = 'runData'
    # File name of the experiments
    fileName = ['ZLC_DeadVolume_Exp16B_Output.mat',
                'ZLC_DeadVolume_Exp16C_Output.mat',
                'ZLC_DeadVolume_Exp16D_Output.mat']

    # Threshold factor (If -negative infinity not used, if not need a float)
    # This is used to split the compositions into two distint regions
    thresholdFactor = -np.inf

    #####################################
    #####################################

    # Save the threshold factor to a dummy file (to pass through GA - IDIOTIC)
    savez ('tempFittingParametersDV.npz',thresholdFactor=thresholdFactor)
    
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,mainDir,fileName,'DV')

    # Define the bounds and the type of the parameters to be optimized                       
    optBounds = np.array(([np.finfo(float).eps,10], [np.finfo(float).eps,10],
                          [np.finfo(float).eps,10], [1,30], [np.finfo(float).eps,0.05]))
                         
    optType=np.array(['real','real','real','int','real'])
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
    model = ga(function = deadVolObjectiveFunction, dimension=5, 
                               variable_type_mixed = optType,
                               variable_boundaries = optBounds,
                               algorithm_parameters=algorithm_param)

    # Call the GA optimizer using multiple cores
    model.run(set_function=ga.set_function_multiprocess(deadVolObjectiveFunction,
                                                         n_jobs = num_cores),
              no_plot = True)
    # Repeat the optimization with the last generation repeated numOptRepeat
    # times (for better accuracy)
    for ii in range(numOptRepeat):
        model.run(set_function=ga.set_function_multiprocess(deadVolObjectiveFunction,
                                                             n_jobs = num_cores),
                  start_generation=model.output_dict['last_generation'], no_plot = True)
    
    # Save the array concentration into a native numpy file
    # The .npz file is saved in a folder called simulationResults (hardcoded)
    filePrefix = "deadVolumeCharacteristics"
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
           mleThreshold = thresholdFactor, # Threshold for MLE composition split [-]
           hostName = socket.gethostname()) # Hostname of the computer) 

    # Remove all the .npy files genereated from the .mat
    # Load the names of the file to be used for estimating dead volume characteristics
    filePath = filesToProcess(False,[],[],'DV')
    # Loop over all available files    
    for ii in range(len(filePath)):
        os.remove(filePath[ii])
        
    # Return the optimized values
    return model.output_dict

# func: deadVolObjectiveFunction
# For use with GA, the function accepts only one input (parameters from the 
# optimizer)
def deadVolObjectiveFunction(x):
    import numpy as np
    from simulateDeadVolume import simulateDeadVolume
    from computeMLEError import computeMLEError
    from numpy import load
    
    # Load the threshold factor from the dummy file
    thresholdFactor = load ('tempFittingParametersDV.npz')["thresholdFactor"]
    
    # Load the names of the file to be used for estimating dead volume characteristics
    filePath = filesToProcess(False,[],[],'DV')
    
    numPointsExp = np.zeros(len(filePath))
    for ii in range(len(filePath)): 
        # Load experimental molefraction
        timeElapsedExp = load(filePath[ii])["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
        
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)
        
    # Initialize error for objective function
    computedError = 0 # Total error
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])
    expVolume = 0
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
        flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                
        # Integration and ode evaluation time (check simulateDeadVolume)
        timeInt = timeElapsedExp

        # Compute the experimental volume (using trapz)
        expVolume = max([expVolume, np.trapz(moleFracExp,np.multiply(flowRateExp, timeElapsedExp))])
        
        # Compute the dead volume response using the optimizer parameters
        _ , _ , moleFracSim = simulateDeadVolume(deadVolume_1 = x[0],
                                                deadVolume_2M = x[1],
                                                deadVolume_2D = x[2],                                      
                                                numTanks_1 = int(x[3]),
                                                flowRate_D = x[4],
                                                timeInt = timeInt,
                                                flowRate = flowRateExp,
                                                expFlag = True)
                            
        # Stack mole fraction from experiments and simulation for error 
        # computation
        moleFracExpALL = np.hstack((moleFracExpALL, moleFracExp))
        moleFracSimALL = np.hstack((moleFracSimALL, moleFracSim))

    # Penalize if the total volume of the system is greater than experiemntal 
    # volume
    penaltyObj = 0
    if sum(x[0:3])>1.5*expVolume:
        penaltyObj = 10000
    # Compute the sum of the error for the difference between exp. and sim. and
    # add a penalty if needed (using MLE)
    computedError = computeMLEError(moleFracExpALL,moleFracSimALL,thresholdFactor=thresholdFactor)
    return computedError + penaltyObj

# func: filesToProcess
# Loads .mat experimental file and processes it for python
def filesToProcess(initFlag,mainDir,fileName,expType):
    from processExpMatFile import processExpMatFile
    from numpy import savez
    from numpy import load
    # Process the data for python (if needed)
    if initFlag:
        savePath=list()
        for ii in range(len(fileName)):
            savePath.append(processExpMatFile(mainDir, fileName[ii]))
        # Save the .npz file names in a dummy file
        dummyFileName = 'tempCreation' + '_' + expType + '.npz'
        savez (dummyFileName, savePath = savePath)
    # Returns the path of the .npz file to be used 
    else:
    # Load the dummy file with file names for processing
        dummyFileName = 'tempCreation' + '_' + expType + '.npz'
        savePath = load (dummyFileName)["savePath"]
        return savePath