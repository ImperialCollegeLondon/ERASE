############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2020
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#
# Purpose:
# Estimates the concentration of the gas mixture using the sensor response
# by minimization of the sum of square error between the "true" and the 
# "estimated" differences in the change of mass of the sensor array
#
# Last modified:
# - 2020-11-09, AK: Changes to initial condition and optimizer bounds
# - 2020-11-05, AK: Introduce keyword argument for custom mole fraction
# - 2020-10-30, AK: Fix to find number of gases
# - 2020-10-22, AK: Change error to relative from absolute, add opt bounds,
#                   input arguments, and initial guess
# - 2020-10-21, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def estimateConcentration(numberOfAdsorbents, numberOfGases, moleFracID, sensorID, **kwargs):
    import numpy as np
    from generateTrueSensorResponse import generateTrueSensorResponse
    from scipy.optimize import basinhopping

    # Total pressure of the gas [Pa]
    pressureTotal = np.array([1.e5]);
    
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    temperature = np.array([298.15]);
    
    # Get the individual sensor reponse for all the given "experimental/test" concentrations
    if 'moleFraction' in kwargs:
        sensorTrueResponse = generateTrueSensorResponse(numberOfAdsorbents,numberOfGases,
                                                    pressureTotal,temperature, moleFraction = kwargs["moleFraction"])
        moleFracID = 0 # Index becomes a scalar quantity
    else:
        sensorTrueResponse = generateTrueSensorResponse(numberOfAdsorbents,numberOfGases,
                                                    pressureTotal,temperature)
        # True mole fraction index (provide the index corresponding to the true
        # experimental mole fraction (0-4, check generateTrueSensorResponse.py)
        moleFracID = moleFracID    

    # Sensor combinations used in the array. This is a [gx1] vector that maps to
    # the sorbent/material ID generated using the 
    # generateHypotheticalAdsorbents.py function
    sensorID = np.array(sensorID)
    
    # Parse out the true sensor response for a sensor array with n number of
    # sensors given by sensorID
    arrayTrueResponse = np.zeros(sensorID.shape[0])
    for ii in range(sensorID.shape[0]):
        arrayTrueResponse[ii] = sensorTrueResponse[sensorID[ii],moleFracID]
    
    # Pack the input parameters/arguments useful to compute the objective
    # function to estimate the mole fraction as a tuple
    inputParameters = (arrayTrueResponse, pressureTotal, temperature, sensorID)
    
    # Minimize an objective function to compute the mole fraction of the feed
    # gas to the sensor
    initialCondition = np.random.uniform(0,1,numberOfGases) # Initial guess
    optBounds = np.tile([0.,1.], (numberOfGases,1)) # BOunding the mole fractions
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
    
    # Unpack the tuple that contains the true response, pressure, temperature,
    # and the sensor identifiers
    arrayTrueResponse, pressureTotal, temperature, sensorID = inputParameters
    
    # Reshape the mole fraction to a row vector for compatibility
    moleFraction = np.array([x]) # This is needed to keep the structure as a row instead of column
    # moleFraction = np.array([x,1.-x]).T # This is needed to keep the structure as a row instead of column

    # Compute the sensor reponse for a given mole fraction input
    arraySimResponse = simulateSensorArray(sensorID, pressureTotal, temperature, moleFraction)
    
    # Compute the sum of the error for the sensor array
    return sum(np.power((arrayTrueResponse - arraySimResponse)/arrayTrueResponse,2))