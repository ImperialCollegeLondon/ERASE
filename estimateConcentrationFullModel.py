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
# Estimates the concentration of the gas mixture using the sensor response
# by minimization of the sum of square error between the "true" and the 
# "estimated" differences in the change of mass of the sensor array as a 
# function of time. Here the full model instead of equilibrium characteristics
# is used
#
# Last modified:
# - 2021-01-25, AK: Modify the objective function from eqbm method
# - 2021-01-22, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def estimateConcentrationFullModel(numberOfGases, sensorID, **kwargs):
    import numpy as np
    from generateTrueSensorResponse import generateTrueSensorResponse
    from scipy.optimize import basinhopping

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
    
    # Sensor combinations used in the array. This is a [gx1] vector that maps to
    # the sorbent/material ID generated using the 
    # generateHypotheticalAdsorbents.py function
    sensorID = np.array(sensorID)
    
    # Parse out the true sensor response from the input (time resolved)
    if 'fullModelResponse' in kwargs:
        arrayTrueResponse = kwargs["fullModelResponse"]
    else:
        errorString = "Sensor response from full model not available. You should not be here!"
        raise Exception(errorString)
        
    # Parse out the rate constant for the materials from the input
    if 'rateConstant' in kwargs:
        rateConstant = kwargs["rateConstant"]
    else:
        errorString = "Rate constant for materials not available. You should not be here!"
        raise Exception(errorString)
        
    # Parse out the time of integration from the input
    if 'timeInt' in kwargs:
        timeInt = kwargs["timeInt"]
    else:
        errorString = "Integration time for the model not available. You should not be here!"
        raise Exception(errorString)
        
    # Parse out the feed flow rate from the input
    if 'flowIn' in kwargs:
        flowIn = kwargs["flowIn"]
    else:
        errorString = "Feed flow rate not available. You should not be here!"
        raise Exception(errorString)

    # Replace all negative values to eps (for physical consistency). Set to 
    # eps to avoid division by zero        
    # Print if any of the responses are negative
    if (arrayTrueResponse<=0.).any():       
        print("Number of zero/negative response: " + str(np.sum(arrayTrueResponse<=0.)))
    arrayTrueResponse[arrayTrueResponse<=0.] = np.finfo(float).eps
    
    # Pack the input parameters/arguments useful to compute the objective
    # function to estimate the mole fraction as a tuple
    inputParameters = (arrayTrueResponse, flowIn, sensorID, rateConstant, timeInt)
    
    # Minimize an objective function to compute the mole fraction of the feed
    # gas to the sensor
    initialCondition = np.random.uniform(0,1,numberOfGases+1) # Initial guess
    optBounds = np.tile([0.,1.], (numberOfGases+1,1)) # BOunding the mole fractions
    optCons = {'type':'eq','fun': lambda x: sum(x) - 1} # Cinstrain the sum to 1.
    # Use the basin hopping minimizer to escape local minima when evaluating
    # the function. The number of iterations is hard-coded and fixed at 50
    estMoleFraction = basinhopping(concObjectiveFunction, initialCondition,
                                    minimizer_kwargs = {"args": inputParameters, 
                                                        "bounds": optBounds,
                                                        "constraints": optCons},
                                    niter = 50)
    return np.concatenate((sensorID,estMoleFraction.x), axis=0)

# func: concObjectiveFunction, computes the sum of square error for the 
# finger print for varying gas concentrations, using the minimize function
def concObjectiveFunction(x, *inputParameters):
    import numpy as np
    from simulateSensorArray import simulateSensorArray
    from simulateFullModel import simulateFullModel
    
    # Unpack the tuple that contains the true response, sensor identifiers, 
    # rate constant and time of integration
    # IMPORTANT: Pressure and Temperature not considered here!!!
    arrayTrueResponse, flowIn, sensorID, rateConstant, timeInt = inputParameters
    
    # Reshape the mole fraction to a row vector for compatibility
    moleFraction = np.array(x) # This is needed to keep the structure as a row instead of column

    # Compute the sensor reponse for a given mole fraction input
    # Calls the full sensor model for each material
    outputStruct = {}
    for ii in range(len(sensorID)):
        timeSim , _ , sensorFingerPrint, _ = simulateFullModel(sensorID = sensorID[ii],
                                                               rateConstant = rateConstant[ii],
                                                               feedMoleFrac = moleFraction,
                                                               timeInt = timeInt,
                                                               flowIn = flowIn)
        outputStruct[ii] = {'timeSim':timeSim,
                            'sensorFingerPrint':sensorFingerPrint} # Check simulateFullModel.py for entries

    # Prepare the simulated response at a given mole fraction to compute the 
    # objective function
    timeSim = []
    timeSim = outputStruct[0]["timeSim"] # Need for array initialization
    arraySimResponse = np.zeros([len(timeSim),len(sensorID)])
    for ii in range(len(sensorID)):
        arraySimResponse[:,ii] = outputStruct[ii]["sensorFingerPrint"]

    # Prepare the objective function for the optimizer
    relError = np.divide((arrayTrueResponse - arraySimResponse), arrayTrueResponse)
    relErrorPow2 = np.power(relError, 2) # Square of relative error
    objFunction = np.sum(relErrorPow2) # Sum over all the sensors and times

    # Return the sum of squares of relative errors
    return objFunction