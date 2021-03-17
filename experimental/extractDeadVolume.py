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
#
# Last modified:
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
    import multiprocessing # For parallel processing
    
    # Find out the total number of cores available for parallel processing
    num_cores = multiprocessing.cpu_count()
    
    # Number of times optimization repeated
    numOptRepeat = 3
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Define the bounds and the type of the parameters to be optimized
    optBounds = np.array(([np.finfo(float).eps,100], [1,20]))
    optType=np.array(['real','int'])
    # Algorithm parameters for GA
    algorithm_param = {'max_num_iteration':25,
                       'population_size':100,
                       'mutation_probability':0.1,
                       'crossover_probability': 0.55,
                       'parents_portion': 0.15,
                       'max_iteration_without_improv':None}

    # Minimize an objective function to compute the dead volume and the number of 
    # tanks for the dead volume using GA
    model = ga(function = deadVolObjectiveFunction, dimension=2, 
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
    
    # Return the optimized values
    return model.output_dict

# func: deadVolObjectiveFunction
# For use with GA, the function accepts only one input (parameters from the 
# optimizer)
def deadVolObjectiveFunction(x):
    import numpy as np
    from simulateDeadVolume import simulateDeadVolume
    
    # Dead volume [cc]
    deadVolume = 3.25
    # Number of tanks [-]
    numberOfTanks = 6    
    # Generate dead volume response (pseudo experiment)
    _ , _ , moleFracExp = simulateDeadVolume(deadVolume = deadVolume,
                                             numberOfTanks = numberOfTanks)
    
    # Compute the dead volume response using the optimizer parameters
    _ , _ , moleFracOut = simulateDeadVolume(deadVolume = x[0],
                                             numberOfTanks = int(x[1]))
    
    # Compute the sum of the error for the difference between exp. and sim.
    return np.sum(np.power(moleFracExp - moleFracOut,2))