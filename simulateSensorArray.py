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
# Generates the response of a sensor array as change in mass due to gas
# sorption at a given pressure, temperature, and mole fraction for 
# n sorbents. 
#
# Last modified:
# - 2021-01-20, AK: Structure change and add material file for full model
# - 2021-01-19, AK: Add flag for full model
# - 2020-10-30, AK: Fix to find number of gases
# - 2020-10-22, AK: Add two/three gases
# - 2020-10-21, AK: Cosmetic changes and make it a function
# - 2020-10-20, AK: Obtain sensor array finger print
# - 2020-10-19, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def simulateSensorArray(sensorID, pressureTotal, temperature, moleFraction, **kwargs):
    import numpy as np
    from numpy import load
    import os
    from simulateSSL import simulateSSL

    # Flag to check if simulation full model or not
    if 'fullModel' in kwargs:
        if kwargs["fullModel"]:
            flagFullModel = True
        else:
            flagFullModel = False
    else:
        flagFullModel = False

    # Load a given adsorbent isotherm material file based on full model flag
    if flagFullModel:
        if moleFraction.shape[1] == 3:
            loadFileName = "isothermParameters_20210120_1722_fb57143.npz" # Two gases + Inert
        elif moleFraction.shape[1] == 4:
            print('yes')
            loadFileName = "isothermParameters_20210120_1724_fb57143.npz" # Three gases + Inert    
    else:
        if moleFraction.shape[1] == 2:
            loadFileName = "isothermParameters_20201020_1756_5f263af.npz" # Two gases
        elif moleFraction.shape[1] == 3:
            loadFileName = "isothermParameters_20201022_1056_782efa3.npz" # Three gases
    hypoAdsorbentFile = os.path.join('inputResources',loadFileName);
    
    # Check if the file with the adsorbent properties exist 
    if os.path.exists(hypoAdsorbentFile):
        loadedFileContent = load(hypoAdsorbentFile)
        adsorbentIsothermTemp = loadedFileContent['adsIsotherm']
        adsorbentDensityTemp = loadedFileContent['adsDensity']
        molecularWeight = loadedFileContent['molWeight']
    else:
        errorString = "Adsorbent property file " + hypoAdsorbentFile + " does not exist."
        raise Exception(errorString)
    
    # Get the equilibrium loading for all the sensors for each gas
    # This is a [nxg] matrix where n is the number of sensors and g the number
    # of gases
    sensorLoadingPerGasVol = np.zeros((sensorID.shape[0],moleFraction.shape[1])) # [mol/m3]
    sensorLoadingPerGasMass = np.zeros((sensorID.shape[0],moleFraction.shape[1])) # [mol/kg]
    for ii in range(sensorID.shape[0]): 
        adsorbentID = sensorID[ii]
        adsorbentIsotherm = adsorbentIsothermTemp[:,:,adsorbentID]
        adsorbentDensity = adsorbentDensityTemp[adsorbentID]
        equilibriumLoadings = simulateSSL(adsorbentIsotherm,adsorbentDensity,
                                          pressureTotal,temperature,moleFraction) # [mol/m3]
        sensorLoadingPerGasVol[ii,:] = equilibriumLoadings[0,0,:] # [mol/m3]
        sensorLoadingPerGasMass[ii,:] = equilibriumLoadings[0,0,:]/adsorbentDensity # [mol/kg]
    
    # Obtain the sensor finger print # [g of total gas adsorbed/kg of sorbent]
    sensorFingerPrint = np.dot(sensorLoadingPerGasMass,molecularWeight) # [g/kg]
    
    # Flag to check if simulation full model or not
    if flagFullModel:
        return sensorLoadingPerGasVol, adsorbentDensity, molecularWeight
    else:
        return sensorFingerPrint