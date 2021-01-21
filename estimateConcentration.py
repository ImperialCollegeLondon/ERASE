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
# - 2021-01-21, AK: Add full model functionality
# - 2020-11-12, AK: Bug fix for multipler error
# - 2020-11-11, AK: Add multiplier error to true sensor response
# - 2020-11-10, AK: Add measurement noise to true sensor response
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

    # Get the individual sensor reponse for all the given "experimental/test" concentrations
    if 'fullModel' in kwargs:
        if kwargs["fullModel"]:
            fullModelFlag = True   
        else:
            fullModelFlag = False
    else:
        fullModelFlag = False
        
    # Add measurement noise for the true measurement if the user wants it
    measurementNoise = np.zeros(sensorID.shape[0])
    if 'addMeasurementNoise' in kwargs:
        # The mean and the standard deviation of the Gaussian error is an 
        # input from the user
        measurementNoise = np.random.normal(kwargs["addMeasurementNoise"][0],
                                            kwargs["addMeasurementNoise"][1],
                                            sensorID.shape[0])
    
    # Check if it is for full model or not
    # Full model condition
    if fullModelFlag:
        # Parse out the true sensor response from the input (time resolved)
        # and add measurement noise if asked for. There is no multipler error for 
        # full model simualtions
        # Note that using the full model here is only for comparison purposes
        # When kinetics are available the other estimateConcentration function
        # should be used
        if 'fullModelResponse' in kwargs:
            fullModelResponse = kwargs["fullModelResponse"]
            multiplierError = np.ones(sensorID.shape[0]) # Always set to 1.
            arrayTrueResponse = np.zeros(sensorID.shape[0])
            for ii in range(sensorID.shape[0]):
                arrayTrueResponse[ii] = (multiplierError[ii]*fullModelResponse[ii] 
                                         + measurementNoise[ii])
        else:
            errorString = "Sensor response from full model not available. You should not be here!"
            raise Exception(errorString)
    # Equilibrium condition
    else:
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
    
        # Add a multiplier error for the true measurement if the user wants it
        multiplierError = np.ones(sensorID.shape[0])
        if 'multiplierError' in kwargs:
            # The mean and the standard deviation of the Gaussian error is an 
            # input from the user
                multiplierErrorTemp = kwargs["multiplierError"]
                multiplierError[0:len(multiplierErrorTemp)] = multiplierErrorTemp
        # Parse out the true sensor response for a sensor array with n number of
        # sensors given by sensorID
        arrayTrueResponse = np.zeros(sensorID.shape[0])
        for ii in range(sensorID.shape[0]):
            arrayTrueResponse[ii] = (multiplierError[ii]*sensorTrueResponse[sensorID[ii],moleFracID]
                                     + measurementNoise[ii])

    # Replace all negative values to eps (for physical consistency). Set to 
    # eps to avoid division by zero        
    # Print if any of the responses are negative
    if any(ii<=0. for ii in arrayTrueResponse):       
        print("Number of negative response: " + str(sum(arrayTrueResponse<0)))
    arrayTrueResponse[arrayTrueResponse<0.] = np.finfo(float).eps
    
    # Pack the input parameters/arguments useful to compute the objective
    # function to estimate the mole fraction as a tuple
    inputParameters = (arrayTrueResponse, pressureTotal, temperature, 
                       sensorID, multiplierError)
    
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
    arrayTrueResponse, pressureTotal, temperature, sensorID, multiplierError = inputParameters
    
    # Reshape the mole fraction to a row vector for compatibility
    moleFraction = np.array([x]) # This is needed to keep the structure as a row instead of column
    # moleFraction = np.array([x,1.-x]).T # This is needed to keep the structure as a row instead of column

    # Compute the sensor reponse for a given mole fraction input
    arraySimResponse = simulateSensorArray(sensorID, pressureTotal, temperature, moleFraction) * multiplierError

    # Compute the sum of the error for the sensor array
    return sum(np.power((arrayTrueResponse - arraySimResponse)/arrayTrueResponse,2))