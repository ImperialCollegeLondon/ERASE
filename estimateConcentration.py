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
# - 2020-10-21, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def estimateConcentration(moleFracID,sensorID):
    import numpy as np
    from generateTrueSensorResponse import generateTrueSensorResponse
    from scipy.optimize import basinhopping
    
    # Total number of sensor elements/gases simulated and generated using 
    # generateHypotheticalAdsorbents.py function
    numberOfAdsorbents = 100;
    numberOfGases = 2;
    
    # Total pressure of the gas [Pa]
    pressureTotal = np.array([1.e5]);
    
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    temperature = np.array([298.15]);
    
    # Get the individual sensor reponse for all the given "experimental/test" concentrations
    sensorTrueResponse = generateTrueSensorResponse(numberOfAdsorbents,pressureTotal,temperature)
    
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
    initialCondition = np.zeros(numberOfGases) # Initial guess
    # Use the basin hopping minimizer to escape local minima when evaluating
    # the function. The number of iterations is hard-coded and fixed at 50
    estMoleFraction = basinhopping(concObjectiveFunction, initialCondition, 
                                   minimizer_kwargs = {"args": inputParameters}, niter = 50)
    return estMoleFraction.x

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
    
    # Compute the sensor reponse for a given mole fraction input
    arraySimResponse = simulateSensorArray(sensorID, pressureTotal, temperature, moleFraction)
    
    # Compute the sum of the error for the sensor array
    return sum(np.power(arrayTrueResponse - arraySimResponse,2))