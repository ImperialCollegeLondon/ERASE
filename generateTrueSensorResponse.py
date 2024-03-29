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
# - 2020-11-09, AK: Introduce custom mole fraction input
# - 2020-11-05, AK: Introduce new case for mole fraction sweep
# - 2020-10-30, AK: Fix to find number of gases
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

def generateTrueSensorResponse(numberOfAdsorbents, numberOfGases, pressureTotal, temperature, **kwargs):
    import numpy as np
    import pdb
    from simulateSensorArray import simulateSensorArray

    # Mole fraction of the gas [-]
    # Can be [jxg], where j is the number of mole fractions for g gases
    if numberOfGases == 2:
        moleFraction = np.array([[0.05, 0.95],
                                  [0.15, 0.85],
                                  [0.40, 0.60],
                                  [0.75, 0.25],
                                  [0.90, 0.10]])
    elif numberOfGases == 3:
        moleFraction = np.array([[0.05, 0.15, 0.80],
                                  [0.15, 0.25, 0.60],
                                  [0.40, 0.35, 0.25],
                                  [0.75, 0.10, 0.15],
                                  [0.90, 0.05, 0.05]])        
    # To sweep through the entire mole fraction range for 2 gases
    elif numberOfGases == 20000:
        moleFraction = np.array([np.linspace(0,1,1001), 1 - np.linspace(0,1,1001)]).T
    # To sweep through the entire mole fraction range for 3 gases
    elif numberOfGases == 30000:
        moleFractionTemp = np.zeros([1001,3])
        num1 = np.random.uniform(0.0,1.0,1001)
        num2 = np.random.uniform(0.0,1.0,1001)
        num3 = np.random.uniform(0.0,1.0,1001)
        sumNum = num1 + num2 + num3
        moleFractionTemp[:,0] = num1/sumNum
        moleFractionTemp[:,1] = num2/sumNum
        moleFractionTemp[:,2] = num3/sumNum
        moleFraction = moleFractionTemp[moleFractionTemp[:,0].argsort()]   
    
    # Check if a custom mole fraction is provided intead of the predefined one
    if 'moleFraction' in kwargs:
        moleFraction = np.array([kwargs["moleFraction"]])
    
    # Get the individual sensor reponse for all the five "test" concentrations
    sensorTrueResponse = np.zeros((numberOfAdsorbents,moleFraction.shape[0]))
    for ii in range(numberOfAdsorbents):
        for jj in range(moleFraction.shape[0]):
            moleFractionTemp = np.array([moleFraction[jj,:]]) # This is needed to keep the structure as a row instead of column
            sensorTrueResponse[ii,jj] = simulateSensorArray(np.array([ii]), 
                                                            pressureTotal, temperature, moleFractionTemp)
    return sensorTrueResponse