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
# Generates a "true" sensor response for known mole fractions of gas for the 
# sorbents that are hypothetically generated. The mole fraction is hard coded
# and should be modified.
#
# Last modified:
# - 2020-10-22, AK: Add two/three gases
# - 2020-10-21, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def generateTrueSensorResponse(numberOfAdsorbents, pressureTotal, temperature):
    import numpy as np
    from simulateSensorArray import simulateSensorArray
    
    # Mole fraction of the gas [-]
    # Can be [jxg], where j is the number of mole fractions for g gases
    # moleFraction = np.array([[0.05, 0.95],
    #                          [0.15, 0.85],
    #                          [0.40, 0.60],
    #                          [0.75, 0.25],
    #                          [0.90, 0.10]])
    moleFraction = np.array([[0.05, 0.15, 0.80],
                              [0.15, 0.25, 0.60],
                              [0.40, 0.35, 0.25],
                              [0.75, 0.10, 0.15],
                              [0.90, 0.05, 0.05]])
    
    # Get the individual sensor reponse for all the five "test" concentrations
    sensorTrueResponse = np.zeros((numberOfAdsorbents,moleFraction.shape[0]))
    for ii in range(numberOfAdsorbents):
        for jj in range(moleFraction.shape[0]):
            moleFractionTemp = np.array([moleFraction[jj,:]]) # This is needed to keep the structure as a row instead of column
            sensorTrueResponse[ii,jj] = simulateSensorArray(np.array([ii]), 
                                                            pressureTotal, temperature, moleFractionTemp)
    return sensorTrueResponse