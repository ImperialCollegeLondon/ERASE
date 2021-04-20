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
    
    # Find out the total number of cores available for parallel processing
    num_cores = multiprocessing.cpu_count()
    
    # Number of times optimization repeated
    numOptRepeat = 10
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Directory of raw data
    mainDir = 'experimental/runData'
    # File name of the experiments
    fileName = ['ZLC_DeadVolume_Exp12D_Output.mat',
                'ZLC_DeadVolume_Exp12E_Output.mat',
                'ZLC_DeadVolume_Exp12F_Output.mat']
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,mainDir,fileName)

    # Define the bounds and the type of the parameters to be optimized                       
    optBounds = np.array(([np.finfo(float).eps,100], [np.finfo(float).eps,100],
                          [np.finfo(float).eps,100], [np.finfo(float).eps,100],
                          [1,30], [1,30], [1,30], [1,30],
                          [np.finfo(float).eps,1-np.finfo(float).eps],
                          [np.finfo(float).eps,1-np.finfo(float).eps]))
                         
    optType=np.array(['real','real','real','real','int','int','int','int','real','real'])
    # Algorithm parameters for GA
    algorithm_param = {'max_num_iteration':10,
                       'population_size':800,
                       'mutation_probability':0.1,
                       'crossover_probability': 0.55,
                       'parents_portion': 0.15,
                       'elit_ratio': 0.01,
                       'max_iteration_without_improv':None}

    # Minimize an objective function to compute the dead volume and the number of 
    # tanks for the dead volume using GA
    model = ga(function = deadVolObjectiveFunction, dimension=10, 
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
           fileName = fileName, # Names of file used for fitting
           hostName = socket.gethostname()) # Hostname of the computer) 
        
    # Return the optimized values
    return model.output_dict

# func: deadVolObjectiveFunction
# For use with GA, the function accepts only one input (parameters from the 
# optimizer)
def deadVolObjectiveFunction(x):
    import numpy as np
    from simulateDeadVolume import simulateDeadVolume
    from numpy import load
    
    # Load the names of the file to be used for estimating dead volume characteristics
    filePath = filesToProcess(False,[],[])
    
    numPointsExp = np.zeros(len(filePath))
    for ii in range(len(filePath)): 
        # Load experimental molefraction
        timeElapsedExp = load(filePath[ii])["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
        
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)
        
    # Initialize error for objective function
    computedError = 0
    numPoints = 0
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
        flowRate = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                
        # Integration and ode evaluation time (check simulateDeadVolume)
        timeInt = timeElapsedExp

        # Compute the experimental volume (using trapz)
        expVolume = max([expVolume, np.trapz(moleFracExp,np.multiply(flowRate, timeElapsedExp))])
        
        # Compute the dead volume response using the optimizer parameters
        _ , _ , moleFracSim = simulateDeadVolume(deadVolume_1M = x[0],
                                                          deadVolume_1D = x[1],
                                                          deadVolume_2M = x[2],
                                                          deadVolume_2D = x[3],
                                                          numTanks_1M = int(x[4]),
                                                          numTanks_1D = int(x[5]),
                                                          numTanks_2M = int(x[6]),
                                                          numTanks_2D = int(x[7]),
                                                          splitRatio_1 = x[8],
                                                          splitRatio_2 = x[9],
                                                          timeInt = timeInt,
                                                          flowRate = flowRate)
                
        # Compute the sum of the error for the difference between exp. and sim.
        numPoints += len(moleFracExp)
        computedError += np.log(np.sum(np.power(moleFracExp - moleFracSim,2)))
    
    # Penalize if the total volume of the system is greater than experiemntal 
    # volume
    penaltyObj = 0
    if sum(x[0:4])>1.5*expVolume:
        penaltyObj = 10000
    # Compute the sum of the error for the difference between exp. and sim. and
    # add a penalty if needed
    return (numPoints/2)*computedError + penaltyObj

# func: filesToProcess
# Loads .mat experimental file and processes it for python
def filesToProcess(initFlag,mainDir,fileName):
    from processExpMatFile import processExpMatFile
    from numpy import savez
    from numpy import load
    # Process the data for python (if needed)
    if initFlag:
        savePath=list()
        for ii in range(len(fileName)):
            savePath.append(processExpMatFile(mainDir, fileName[ii]))
        # Save the .npz file names in a dummy file
        savez ('tempCreation.npz', savePath = savePath)
    # Returns the path of the .npz file to be used 
    else:
    # Load the dummy file with file names for processing
        savePath = load ('tempCreation.npz')["savePath"]
        return savePath